from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
import jieba

from legalrag.config import AppConfig
from legalrag.models import LawChunk
from legalrag.utils.logger import get_logger
from legalrag.retrieval.corpus_loader import load_chunks_from_dir


logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval
        self.bm25_path = Path(rcfg.bm25_index_file)
        self.processed_path = Path(rcfg.processed_file)
        self.processed_dir = rcfg.processed_dir
        self.processed_glob = getattr(rcfg, "processed_glob", "*.jsonl")

        self.bm25: BM25Okapi | None = None
        self.chunks: List[LawChunk] = []
        self.corpus_tokens: List[List[str]] = []

    def build(self):
        logger.info("[BM25] 构建索引（扫描 processed_dir/*.jsonl）")

        processed_dir = self.cfg.paths.processed_dir  # 与 PDFIngestor 输出对齐 :contentReference[oaicite:2]{index=2}
        self.chunks = load_chunks_from_dir(processed_dir, pattern="*.jsonl")

        self.corpus_tokens = [list(jieba.cut(c.text)) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with self.bm25_path.open("wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

        logger.info(f"[BM25] saved -> {self.bm25_path} (docs={len(self.chunks)})")

    def load(self):
        if self.bm25 is not None:
            return
        if not self.bm25_path.exists():
            self.build()
            return
        with self.bm25_path.open("rb") as f:
            obj = pickle.load(f)
        self.bm25 = obj["bm25"]
        self.chunks = obj["chunks"]
        logger.info(f"[BM25] Loaded index with {len(self.chunks)} chunks")

    def search(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        self.load()
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = [(self.chunks[i], float(scores[i])) for i in idxs]
        return hits
