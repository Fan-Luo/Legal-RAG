from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk, RetrievalHit
from legalrag.retrieval.vector_store import VectorStore
from legalrag.retrieval.graph_store import LawGraphStore
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1:
        a = a.reshape(-1)
    if b.ndim != 1:
        b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _extract_key_term(question: str) -> str:
    import re
    toks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", question or "")
    if not toks:
        return (question or "").strip()[:12]
    return max(toks, key=len)


@dataclass
class GraphRetriever:
    """
    Graph-aware retriever
    - Seeds from other channels provide starting nodes.
    - Walk graph to collect candidate nodes.
    - Score candidates by semantic similarity (VectorStore embedder).
    - Return graph_hits  
    """

    cfg: AppConfig
    graph: Optional[LawGraphStore] = None
    store: Optional[VectorStore] = None

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = LawGraphStore(self.cfg)
        if self.store is None:
            self.store = VectorStore(self.cfg)

    def search(
        self,
        question: str,
        seed_hits: List[RetrievalHit],
        *,
        decision: Any = None,
        top_k: int = 50,
        depth: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[RetrievalHit]:
        if not seed_hits:
            return []

        assert self.graph is not None
        assert self.store is not None

        rcfg = getattr(self.cfg, "retrieval", None)
        if depth is None:
            depth = int(getattr(rcfg, "graph_depth", 2)) if rcfg else 2
        if limit is None:
            limit = int(getattr(rcfg, "graph_limit", 120)) if rcfg else 120

        # Seeds -> start_ids
        seed_ids: List[str] = []
        for h in seed_hits:
            sid = getattr(h.chunk, "article_id", None) or getattr(h.chunk, "id", None)
            if sid:
                seed_ids.append(str(sid))
        seed_ids = list(dict.fromkeys(seed_ids))
        if not seed_ids:
            return []

        # Walk graph
        try:
            nodes = self.graph.walk(
                start_ids=seed_ids,
                depth=int(depth),
                rel_types=getattr(rcfg, "graph_rel_types", None) if rcfg else None,
                limit=int(limit),
            )
        except Exception:
            logger.warning("[graph] walk failed", exc_info=True)
            nodes = []

        # Definition-specific expansion
        qtype = getattr(decision, "query_type", None)
        if str(qtype).upper().endswith("DEFINITION"):
            term = _extract_key_term(question)
            try:
                nodes.extend(self.graph.get_definition_sources(term))
            except Exception:
                logger.warning("[graph] get_definition_sources failed", exc_info=True)

        # Dedup + filter out seed ids
        seen = set(seed_ids)
        uniq_nodes: List[Any] = []
        for n in nodes:
            aid = getattr(n, "article_id", None) or getattr(n, "id", None)
            if not aid:
                continue
            aid = str(aid)
            if aid in seen:
                continue
            seen.add(aid)
            uniq_nodes.append(n)

        if not uniq_nodes:
            return []

        # Nodes -> chunks
        graph_chunks: List[LawChunk] = []
        meta: List[dict] = []
        for n in uniq_nodes:
            try:
                article_id = str(getattr(n, "article_id", None) or getattr(n, "id", ""))
                title = getattr(n, "title", None) or getattr(n, "article_title", None) or ""
                chapter = getattr(n, "chapter", None) or getattr(n, "section", None) or ""
                content = getattr(n, "text", None) or getattr(n, "content", None) or ""
                text = f"{title}\n{content}".strip() if title else str(content).strip()
                if not article_id or not text:
                    continue
                graph_chunks.append(
                    LawChunk(
                        id=article_id,
                        article_id=article_id,
                        title=str(title) if title else None,
                        chapter=str(chapter) if chapter else None,
                        text=text,
                        source="graph",
                    )
                )
                meta.append(
                    {
                        "graph_depth": int(getattr(n, "graph_depth", 1) or 1),
                        "relations": list(getattr(n, "relations", []) or []),
                    }
                )
            except Exception:
                continue

        if not graph_chunks:
            return []

        # Semantic scoring
        self.store.load()
        qvec = self.store._embed([question]).astype("float32")[0]
        doc_vecs = self.store._embed([c.text for c in graph_chunks]).astype("float32")

        hits: List[RetrievalHit] = []
        for i, (c, v, m) in enumerate(zip(graph_chunks, doc_vecs, meta), start=1):
            sem = _cosine_sim(qvec, v)
            hits.append(
                RetrievalHit(
                    chunk=c,
                    score=float(sem),
                    rank=i,
                    source="graph",
                    semantic_score=float(sem),
                    graph_depth=int(m["graph_depth"]),
                    relations=m["relations"],
                    seed_article_id=seed_ids[0] if seed_ids else None,
                    score_breakdown={
                        "channel": "graph",
                        "semantic": float(sem),
                        "graph_depth": int(m["graph_depth"]),
                        "relations": m["relations"],
                    },
                )
            )

        hits.sort(key=lambda x: float(x.score), reverse=True)
        hits = hits[: max(1, int(top_k))]
        for r, h in enumerate(hits, start=1):
            h.rank = r
        return hits
