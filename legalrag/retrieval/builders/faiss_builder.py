from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed_batch(texts: List[str], model_name: str, device: torch.device, batch_size: int = 64) -> np.ndarray:
    """
    Offline embedding with the configured embedding model (e.g., BGE).
    Returns L2-normalized numpy vectors.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embs = pooled.cpu().numpy().astype("float32")
            all_embs.append(embs)

    emb = np.concatenate(all_embs, axis=0)
    faiss.normalize_L2(emb)
    return emb


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
    emb = _embed_batch(texts, rcfg.embedding_model, device)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype("float32"))

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
