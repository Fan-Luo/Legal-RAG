from __future__ import annotations

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

from legalrag.config import AppConfig
from legalrag.models import LawChunk
from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def load_chunks(path: Path) -> list[LawChunk]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(LawChunk(**obj))
    return chunks


def build_faiss(cfg: AppConfig, chunks: list[LawChunk]):
    rcfg = cfg.retrieval
    vs_model = SentenceTransformer(rcfg.embedding_model)
    texts = [c.text for c in chunks]
    logger.info("[FAISS] 计算向量中...")
    emb = vs_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
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
            f.write(c.model_dump_json(ensure_ascii=False) + "\n")
    logger.info(f"[FAISS] 元数据写入 {meta_path}")


def main():
    cfg = AppConfig.load()
    rcfg = cfg.retrieval
    processed = Path(rcfg.processed_file)
    logger.info(f"Loading processed law from {processed}")
    chunks = load_chunks(processed)
    logger.info(f"Loaded {len(chunks)} law chunks")

    build_faiss(cfg, chunks)

    bm = BM25Retriever(cfg)
    bm.build()


if __name__ == "__main__":
    main()
