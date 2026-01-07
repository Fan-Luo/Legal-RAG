from __future__ import annotations

from pathlib import Path
from typing import List
from FlagEmbedding import FlagModel
import faiss
import numpy as np
import torch
# from transformers import AutoModel, AutoTokenizer

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

_embedding_cache = {}


# def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
#     summed = (last_hidden_state * mask).sum(dim=1)
#     counts = mask.sum(dim=1).clamp(min=1e-9)
#     return summed / counts

def _get_embedder(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    使用 FlagModel 加载嵌入模型（支持 fp16 加速 + instruction）
    """
    if model_name not in _embedding_cache:
        # FlagModel 内部处理 tokenizer + model + instruction
        model = FlagModel(
            model_name,
            query_instruction_for_retrieval="为这个法律问题生成表示以用于检索相关法律条文：",
            use_fp16=torch.cuda.is_available(),    
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        _embedding_cache[model_name] = model
        logger.info(f"[FAISS] Loaded FlagModel for {model_name} (fp16: {model.use_fp16})")
    
    return _embedding_cache[model_name]


def _embed_batch(texts: List[str], model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", batch_size: int = 64) -> np.ndarray:
    """
    使用 FlagModel.encode() 批量编码 passage（建索引时不加 instruction）
    """
    if not texts:
        return np.zeros((0, 768), dtype="float32")  # 假设 dim=768，根据模型调整

    model = _get_embedder(model_name, device)
    
    # FlagModel.encode() 内部已 L2 normalize + 支持 batch
    # passage 使用 encode()，不加 instruction
    embs = model.encode(
        texts,
        batch_size=batch_size,
        max_length=512
    )
    
    # FlagModel 已 normalize，无需再 faiss.normalize_L2(emb)
    return embs.astype("float32")

def build_faiss_index(cfg: AppConfig, chunks: List[LawChunk]) -> None:
    """
    Build FAISS dense index and persist to:
      - cfg.retrieval.faiss_index_file
      - cfg.retrieval.faiss_meta_file (jsonl of LawChunk)
    """
    rcfg = cfg.retrieval
    texts = [c.text for c in chunks]

    logger.info("[FAISS] embedding model=%s", rcfg.embedding_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = _embed_batch(texts, rcfg.embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")

    dim = emb.shape[1]
    # 使用 HNSW  
    hnsw_m = getattr(rcfg, "hnsw_m", 32)
    ef_construction = getattr(rcfg, "hnsw_ef_construction", 200)
    ef_search = getattr(rcfg, "hnsw_ef_search", 128)
    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    # efSearch 可以在查询时动态设置，这里设一个合理的默认值
    index.hnsw.efSearch = ef_search
    logger.info(
        f"[FAISS] Building IndexHNSWFlat (M={hnsw_m}, efConstruction={ef_construction}, efSearch={ef_search})"
    )
    index.add(emb)


    index_path = Path(rcfg.faiss_index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("[FAISS] index written: %s", index_path)

    meta_path = Path(rcfg.faiss_meta_file)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    logger.info("[FAISS] meta written: %s", meta_path)
