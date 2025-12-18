from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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
    - 使用 FAISS IndexFlatIP 做内积检索（向量已 L2 归一化 → 相当于 cosine）
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval

        self.index_path = Path(rcfg.faiss_index_file)
        self.meta_path = Path(rcfg.faiss_meta_file)

        model_name = rcfg.embedding_model
        logger.info(f"[VectorStore] Loading embedding model (BGE): {model_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[LawChunk] = []


    def load(self):
        """
        加载 FAISS 索引 + 元数据 JSONL → LawChunk 列表。
        """
        if self.index is not None and self.chunks:
            return

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS 索引或元数据不存在，请先运行 scripts.build_index.")

        self.index = faiss.read_index(str(self.index_path))
        self.chunks = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(LawChunk(**obj))

        logger.info(f"[FAISS] Loaded {len(self.chunks)} chunks")


    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        使用 BGE-base 做句向量编码：
        - tokenizer → model → mean pooling
        - L2 归一化 → float32 numpy
        """
        if not texts:
            return np.zeros((0, 768), dtype="float32")

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = pooled.cpu().numpy().astype("float32")

        faiss.normalize_L2(emb)
        return emb


    def search(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        """
        使用 dense 向量（BGE）在 FAISS 上做内积检索。
        """
        self.load()
        q_vec = self._embed([query])  # [1, dim]
        scores, idxs = self.index.search(q_vec, top_k)

        hits: List[Tuple[LawChunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hits.append((self.chunks[idx], float(score)))
        return hits


        
