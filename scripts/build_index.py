from __future__ import annotations

import argparse

import faiss  
import numpy as np

from legalrag.config import AppConfig
from legalrag.retrieval.builders.bm25_builder import build_bm25_index
from legalrag.retrieval.builders.colbert_builder import build_colbert_index
from legalrag.retrieval.builders.faiss_builder import build_faiss_index
from legalrag.retrieval.corpus_loader import load_chunks_from_dir
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build indices for LegalRAG (FAISS dense + BM25 sparse + optional ColBERT)."
    )
    g = p.add_mutually_exclusive_group()

    p.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW parameter M (higher → better recall, more memory).",
    )
    p.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="HNSW efConstruction (higher → better graph quality, slower build).",
    )
    p.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=128,
        help="HNSW efSearch at query time (higher → higher recall, slower search).",
    )

    g.add_argument("--only-colbert", action="store_true", help="Only build ColBERT index (skip FAISS/BM25).")
    g.add_argument("--only-faiss", action="store_true", help="Only build FAISS (skip BM25/ColBERT).")
    g.add_argument("--only-bm25", action="store_true", help="Only build BM25 (skip FAISS/ColBERT).")
    p.add_argument("--no-faiss", action="store_true", help="Skip FAISS in default build.")
    p.add_argument("--no-bm25", action="store_true", help="Skip BM25 in default build.")
    p.add_argument("--no-colbert", action="store_true", help="Skip ColBERT.")


    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig.load()
    rcfg = cfg.retrieval

    rcfg.hnsw_m = args.hnsw_m
    rcfg.hnsw_ef_construction = args.hnsw_ef_construction
    rcfg.hnsw_ef_search = args.hnsw_ef_search

    chunks = load_chunks_from_dir(rcfg.processed_dir, rcfg.processed_glob)
    logger.info("Loaded %d law chunks from %s/%s", len(chunks), rcfg.processed_dir, rcfg.processed_glob)

    if args.only_colbert:
        build_colbert_index(cfg, chunks)
        return

    if args.only_faiss:
        build_faiss_index(cfg, chunks)
        return

    if args.only_bm25:
        build_bm25_index(cfg, chunks)
        return

    if not args.no_faiss:
        build_faiss_index(cfg, chunks)
    if not args.no_bm25:
        build_bm25_index(cfg, chunks)
    if not args.no_colbert:
        try:
            build_colbert_index(cfg, chunks)
        except Exception as e:
            print(f"⚠️ Warning: ColBERT index build failed, continuing without it.\nReason: {e}")

if __name__ == "__main__":
    main()