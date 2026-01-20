from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    Sparse retriever (BM25).
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval
        self.bm25_path = Path(rcfg.bm25_index_file)

        self._loaded = False
        self._bm25_mtime: float | None = None
        self.bm25: BM25Okapi | None = None
        self.chunks: List[LawChunk] = []

    def load(self) -> None:
        if not self.bm25_path.exists():
            raise RuntimeError(
                f"[BM25] index not found: {self.bm25_path}. "
                f"Run: python build_index.py (or python build_index.py --only-bm25)"
            )
        current_mtime = self.bm25_path.stat().st_mtime
        if self._loaded and self._bm25_mtime == current_mtime:
            return

        with self.bm25_path.open("rb") as f:
            obj = pickle.load(f)

        bm25 = obj.get("bm25")
        raw_chunks = obj.get("chunks", [])

        if bm25 is None:
            raise RuntimeError(f"[BM25] invalid index file (missing 'bm25'): {self.bm25_path}")

        chunks: List[LawChunk] = []
        for c in raw_chunks:
            if isinstance(c, LawChunk):
                chunks.append(c)
            elif isinstance(c, dict):
                chunks.append(LawChunk(**c))
            else:
                try:
                    chunks.append(LawChunk(**c.model_dump()))
                except Exception as e:
                    raise RuntimeError(f"[BM25] unsupported chunk format in index: {type(c)}") from e

        self.bm25 = bm25
        self.chunks = chunks
        self._loaded = True
        self._bm25_mtime = current_mtime
        logger.info("[BM25] loaded index with %d chunks", len(self.chunks))

    def search(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        self.load()
        assert self.bm25 is not None

        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: int(top_k)]
        return [(self.chunks[i], float(scores[i])) for i in idxs]
