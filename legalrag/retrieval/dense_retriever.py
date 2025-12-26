from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from legalrag.config import AppConfig
from legalrag.schemas import RetrievalHit
from legalrag.retrieval.vector_store import VectorStore


@dataclass
class DenseRetriever:
    """
    Dense retriever (FAISS + embedding)  

    Responsibilities:
      - Load VectorStore (FAISS index + meta)
      - Embed query
      - Perform ANN search
      - Return RetrievalHit list

    Notes:
      - VectorStore is the "store" layer: embed + load + persistence.
      - This retriever is intentionally thin and testable.
    """
    cfg: AppConfig
    store: Optional[VectorStore] = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = VectorStore(self.cfg)

    def search(self, query: str, top_k: int) -> List[RetrievalHit]:
        assert self.store is not None
        self.store.load()

        k = max(1, int(top_k))
        qvec = self.store._embed([query]).astype("float32")  # shape (1, dim)

        # VectorStore uses IndexFlatIP; assume vectors are normalized => IP ~= cosine
        scores, idxs = self.store.index.search(qvec, k)  # type: ignore[attr-defined]
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        hits: List[RetrievalHit] = []
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            if i < 0 or i >= len(self.store.chunks):
                continue
            chunk = self.store.chunks[i]
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=float(s),
                    rank=rank,
                    source="retriever",
                    semantic_score=float(s),
                )
            )
        return hits
