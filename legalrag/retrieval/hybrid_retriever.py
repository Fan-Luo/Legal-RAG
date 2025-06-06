from __future__ import annotations

from typing import List

from legalrag.config import AppConfig
from legalrag.models import RetrievalHit
from legalrag.retrieval.vector_store import VectorStore
from legalrag.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.vs = VectorStore(cfg)
        self.bm25 = BM25Retriever(cfg)

    def search(self, query: str, top_k: int | None = None) -> List[RetrievalHit]:
        rcfg = self.cfg.retrieval
        k = top_k or rcfg.top_k
        dense_hits = self.vs.search(query, k)
        sparse_hits = self.bm25.search(query, k)

        scores = {}
        source = {}

        for chunk, s in dense_hits:
            scores.setdefault(chunk.id, 0.0)
            scores[chunk.id] += rcfg.dense_weight * s
            source[chunk.id] = chunk

        for chunk, s in sparse_hits:
            scores.setdefault(chunk.id, 0.0)
            scores[chunk.id] += rcfg.bm25_weight * s
            source[chunk.id] = chunk

        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        hits: List[RetrievalHit] = []
        for rank, (cid, score) in enumerate(items, start=1):
            hits.append(
                RetrievalHit(
                    chunk=source[cid],
                    score=float(score),
                    rank=rank,
                )
            )
        return hits
