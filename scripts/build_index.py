from __future__ import annotations

import argparse

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

    # Requested flags
    p.add_argument(
        "--with-colbert",
        action="store_true",
        help="Build FAISS + BM25, and also build ColBERT (late-interaction) index.",
    )
    g.add_argument(
        "--only-colbert",
        action="store_true",
        help="Only build ColBERT index (skip FAISS/BM25).",
    )

    # Additional toggles (useful in CI/experiments)
    g.add_argument("--only-faiss", action="store_true", help="Only build FAISS (skip BM25/ColBERT).")
    g.add_argument("--only-bm25", action="store_true", help="Only build BM25 (skip FAISS/ColBERT).")

    p.add_argument("--no-faiss", action="store_true", help="Skip FAISS in default build.")
    p.add_argument("--no-bm25", action="store_true", help="Skip BM25 in default build.")
    p.add_argument("--no-colbert", action="store_true", help="Skip ColBERT even with --with-colbert.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig.load()
    rcfg = cfg.retrieval

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

    # Default: FAISS + BM25, optional ColBERT
    if not args.no_faiss:
        build_faiss_index(cfg, chunks)

    if not args.no_bm25:
        build_bm25_index(cfg, chunks)

    if args.with_colbert and not args.no_colbert:
        build_colbert_index(cfg, chunks)


if __name__ == "__main__":
    main()
