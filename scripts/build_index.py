from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.utils.logger import get_logger
from legalrag.retrieval.corpus_loader import load_chunks_from_dir

logger = get_logger(__name__)


def load_chunks(path: Path) -> List[LawChunk]:
    chunks: List[LawChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(LawChunk(**obj))
    return chunks


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _embed_batch(texts: List[str], model_name: str, device: torch.device) -> np.ndarray:
    """
    使用 BGE-base 对一批条文进行编码，返回 L2 归一化后的 numpy 向量。
    这里只做一次性编码（离线构建索引）。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embs: List[np.ndarray] = []

    batch_size = 64
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


def build_faiss(cfg: AppConfig, chunks: List[LawChunk]):
    rcfg = cfg.retrieval
    texts = [c.text for c in chunks]

    logger.info("[FAISS] 使用 BGE 计算向量中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = _embed_batch(texts, rcfg.embedding_model, device)  # [N, dim]

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype("float32"))

    index_path = Path(rcfg.faiss_index_file)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(f"[FAISS] 索引写入 {index_path}")

    meta_path = Path(rcfg.faiss_meta_file)
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    logger.info(f"[FAISS] 元数据写入 {meta_path}")


def main():
    cfg = AppConfig.load()
    rcfg = cfg.retrieval
    chunks = load_chunks_from_dir(rcfg.processed_dir, rcfg.processed_glob)
    logger.info(f"Loaded {len(chunks)} law chunks from {rcfg.processed_dir}/{rcfg.processed_glob}")

    # dense：BGE + FAISS
    build_faiss(cfg, chunks)

    # sparse：BM25（Lexical）
    bm = BM25Retriever(cfg)
    bm.build()


if __name__ == "__main__":
    main()
