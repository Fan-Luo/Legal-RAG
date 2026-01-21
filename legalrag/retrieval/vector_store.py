from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple

import faiss
import numpy as np
import os
import torch
# from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import FlagModel
from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    BGE 官方推荐的平均池化方式：
    对每个 token embedding 按 attention_mask 做加权平均。
    """
    # last_hidden_state: [batch, seq_len, hidden]
    # attention_mask: [batch, seq_len]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [b, l, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [b, hidden]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [b, 1]
    return summed / counts

class VectorStore:
    """
    Dense 向量检索后端：
    - 使用 BAAI/bge-base-zh-v1.5（或 cfg.retrieval.embedding_model）作为 encoder
    - 使用 FAISS IndexHNSWFlat 做高效近似内积检索（向量已 L2 归一化 → 相当于 cosine）
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval

        self.index_path = Path(rcfg.faiss_index_file)
        self.meta_path = Path(rcfg.faiss_meta_file)

        model_name = rcfg.embedding_model
        # logger.info(f"[VectorStore] Loading embedding model (BGE): {model_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ask FlagEmbedding to avoid multi-process encoding.
        # os.environ.setdefault("FLAGEMBEDDING_USE_MULTI_PROCESS", "0")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = FlagModel(
            model_name,
            query_instruction_for_retrieval="为这个法律问题生成表示以用于检索相关法律条文：",
            use_fp16=torch.cuda.is_available(),         
            device=self.device             
        )
        # self.model.to(self.device)
        # self.model.eval()
        self.index: faiss.Index | None = None  
        self.chunks: List[LawChunk] = []
        self._index_mtime: float | None = None
        self._meta_mtime: float | None = None

    _instances_by_key: ClassVar[Dict[Tuple[str, str, str, str], "VectorStore"]] = {}

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "VectorStore":
        rcfg = cfg.retrieval
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = (
            str(rcfg.embedding_model),
            str(rcfg.faiss_index_file),
            str(rcfg.faiss_meta_file),
            device,
        )
        if key in cls._instances_by_key:
            return cls._instances_by_key[key]
        inst = cls(cfg)
        cls._instances_by_key[key] = inst
        return inst

    def load(self):
        """
        加载 FAISS HNSW 索引 + 元数据 JSONL → LawChunk 列表。
        """
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS 索引或元数据不存在，请先运行 scripts.build_index.")

        index_mtime = self.index_path.stat().st_mtime
        meta_mtime = self.meta_path.stat().st_mtime
        if (
            self.index is not None
            and self.chunks
            and self._index_mtime == index_mtime
            and self._meta_mtime == meta_mtime
        ):
            return

        self.index = faiss.read_index(str(self.index_path))

        # 设置搜索时的 efSearch 参数（提高召回率）
        if hasattr(self.index, 'hnsw'):
            ef_search = getattr(self.cfg.retrieval, "hnsw_ef_search", 128)
            self.index.hnsw.efSearch = ef_search
            # logger.info(f"[FAISS] Set hnsw.efSearch = {ef_search}")

        self.chunks = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  
                    obj = json.loads(line)
                    self.chunks.append(LawChunk.model_validate(obj))  

        self._index_mtime = index_mtime
        self._meta_mtime = meta_mtime
        # logger.info(f"[FAISS] Loaded {len(self.chunks)} chunks, index type: {type(self.index)}")

    def _embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        使用 FlagModel 进行批量嵌入：
        - is_query=True: 使用 encode_queries（加 instruction）
        - is_query=False: 使用 encode（passage，不加 instruction）
        """
        if not texts:
            dim = self.model.model.config.hidden_size  # 动态获取维度（通常768）
            return np.zeros((0, dim), dtype="float32")

        if is_query:
            embs = self.model.encode_queries(
                texts,
                batch_size=64,
                max_length=512,
            )
        else:
            embs = self.model.encode(
                texts,
                batch_size=64,
                max_length=512,
            )

        # FlagModel 已归一化，无需再 faiss.normalize_L2
        return embs.astype("float32")

    def search(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        """
        使用 dense 向量在 FAISS 上做内积检索（query 使用 instruction）
        """
        self.load()
        q_vec = self._embed([query], is_query=True)  # ← 关键：query 加 instruction

        # 临时提高 efSearch 以提升召回 
        # if hasattr(self.index, 'hnsw'):
        #     original_ef = self.index.hnsw.efSearch
        #     self.index.hnsw.efSearch = 1024

        scores, idxs = self.index.search(q_vec, top_k)

        # if hasattr(self.index, 'hnsw'):
        #     self.index.hnsw.efSearch = original_ef

        hits: List[Tuple[LawChunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hits.append((self.chunks[idx], float(score)))

        return hits
        
