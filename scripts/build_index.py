from __future__ import annotations

import argparse
from pathlib import Path

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
    p.add_argument(
        "--index-version",
        type=str,
        default="",
        help="Write indexes into data/index/versions/<version> and activate if requested.",
    )
    p.add_argument(
        "--activate",
        action="store_true",
        help="Activate the provided --index-version after building.",
    )


    return p.parse_args()


def main() -> None:
    args = parse_args()
    idx_version = (args.index_version or "").strip() or None
    cfg = AppConfig.load(index_version=idx_version)
    rcfg = cfg.retrieval

    rcfg.hnsw_m = args.hnsw_m
    rcfg.hnsw_ef_construction = args.hnsw_ef_construction
    rcfg.hnsw_ef_search = args.hnsw_ef_search

    chunks = load_chunks_from_dir(rcfg.processed_dir, rcfg.processed_glob)
    logger.info("Loaded %d law chunks from %s/%s", len(chunks), rcfg.processed_dir, rcfg.processed_glob)

    by_lang = {}
    for c in chunks:
        lang = (getattr(c, "lang", None) or "zh").strip().lower()
        by_lang.setdefault(lang, []).append(c)

    if not by_lang:
        logger.error("No chunks found to index.")
        return

    for lang, lang_chunks in sorted(by_lang.items()):
        lang_cfg = cfg.with_lang(lang)
        logger.info("Building indexes for lang=%s (chunks=%d)", lang, len(lang_chunks))

        if args.only_colbert:
            build_colbert_index(lang_cfg, lang_chunks)
            continue

        if args.only_faiss:
            build_faiss_index(lang_cfg, lang_chunks)
            continue

        if args.only_bm25:
            build_bm25_index(lang_cfg, lang_chunks)
            continue

        if not args.no_faiss:
            build_faiss_index(lang_cfg, lang_chunks)
        if not args.no_bm25:
            build_bm25_index(lang_cfg, lang_chunks)
        if not args.no_colbert:
            try:
                build_colbert_index(lang_cfg, lang_chunks)
            except Exception as e:
                print(f"⚠️ Warning: ColBERT index build failed for lang={lang}, continuing without it.\nReason: {e}")

    if idx_version and args.activate:
        from legalrag.index.registry import IndexRegistry
        for lang in sorted(by_lang.keys()):
            lang_cfg = cfg.with_lang(lang)
            registry = IndexRegistry(Path(lang_cfg.paths.index_dir))
            registry.activate(idx_version)

if __name__ == "__main__":
    main()
