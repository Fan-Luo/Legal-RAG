from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk, RetrievalHit
from legalrag.retrieval.vector_store import VectorStore
from legalrag.retrieval.graph_store import LawGraphStore
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _depth_decay(depth: int, gamma: float = 0.7) -> float:
    d = max(1, int(depth or 1))
    return float(1.0 / ((1.0 + d) ** gamma))


def _relation_weight(relations: List[str]) -> float:
    rels = [str(r).lower() for r in (relations or [])]
    if not rels:
        return 1.0
    wmap = {
        "defined_by": 1.20,
        "defines_term": 1.10, 
        "cite": 1.15,
        "cited": 1.15,
        "ref": 1.15,
        "amend": 1.10,
        "next": 0.95,
        "prev": 0.95,
        "neighbor": 1.00,
    }
    return float(max(wmap.get(r, 1.0) for r in rels))


def _make_hit(payload: Dict[str, Any]) -> RetrievalHit:

    if hasattr(RetrievalHit, "model_validate"):
        return RetrievalHit.model_validate(payload)  # type: ignore[attr-defined]
    return RetrievalHit(**payload)


@dataclass
class GraphRetriever:
    """
    Graph-aware retriever:
      - start_ids from seeds (other channels)
      - walk graph to collect candidate nodes
      - hydrate LawChunk via VectorStore.id2chunk
      - score by semantic similarity, adjusted by graph_depth / relations / edge conf
    """

    cfg: AppConfig
    graph: Optional[LawGraphStore] = None
    store: Optional[VectorStore] = None
    id2chunk: Optional[Dict[str, LawChunk]] = None

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = LawGraphStore(self.cfg)
        if self.store is None:
            self.store = VectorStore.from_config(self.cfg)

        # Load vector store & build id2chunk
        self.store.load()
        self.id2chunk = {}
        for c in getattr(self.store, "chunks", []) or []:
            aid = getattr(c, "article_id", None) or getattr(c, "id", None)
            if aid:
                self.id2chunk[str(aid)] = c

    def search(
        self,
        question: str,
        seeds: List[Any],
        *,
        decision: Any = None,
        top_k: int = 10,
    ) -> List[RetrievalHit]:
        """
        seeds: list[RetrievalHit] from other channels (dense/bm25/colbert)
        Returns: list[RetrievalHit] with source="graph"
        """
        assert self.graph is not None
        assert self.store is not None
        assert self.id2chunk is not None

        rcfg = getattr(self.cfg, "retrieval", None)
        eff_top_k = max(1, int(top_k))

        # walk params
        relation_depths = rcfg.graph_walk_depths if hasattr(rcfg, "graph_walk_depths") else {"default": 2}
        limit = int(getattr(rcfg, "graph_limit", eff_top_k * 8) if rcfg else eff_top_k * 8)
        rel_types = getattr(rcfg, "graph_rel_types", None) if rcfg else None
        min_conf = float(getattr(rcfg, "graph_min_conf", 0.0) if rcfg else 0.0)
        gamma = float(getattr(rcfg, "graph_depth_gamma", 0.7) if rcfg else 0.7)

        # seed ids
        seed_ids: List[str] = []
        for h in seeds or []:
            c = getattr(h, "chunk", None)
            if c is None:
                continue
            aid = getattr(c, "article_id", None) or getattr(c, "id", None)
            if aid:
                seed_ids.append(str(aid))
        seed_ids = [x for x in seed_ids if x]
        if not seed_ids:
            return []

        # Walk graph 
        nodes = self.graph.walk(
            start_ids=seed_ids,
            relation_max_depth=relation_depths,
            limit=limit,
            rel_types=rel_types,
            min_conf=min_conf,
        )
        if not nodes:
            return []

        # De-dup by article_id
        uniq = {}
        for n in nodes:
            aid = str(getattr(n, "article_id", "") or "").strip()
            if not aid:
                continue
            if aid not in uniq:
                uniq[aid] = n
        uniq_nodes = list(uniq.values())

        # Hydrate chunks + keep meta
        graph_chunks: List[LawChunk] = []
        meta: List[Dict[str, Any]] = []

        for n in uniq_nodes:
            aid = str(getattr(n, "article_id", "") or "").strip()
            if not aid:
                continue
            c = self.id2chunk.get(aid)
            if not c or not (getattr(c, "text", "") or "").strip():
                continue

            cc = copy.copy(c)
            setattr(cc, "source", "graph")
            graph_chunks.append(cc)

            edge_conf = float(((getattr(n, "meta", {}) or {}).get("_edge_conf", 1.0)) or 1.0)
            rels = list(getattr(n, "relations", []) or [])
            meta.append(
                {
                    "graph_depth": int(getattr(n, "graph_depth", 1) or 1),
                    "relations": rels,
                    "edge_conf": edge_conf,
                }
            )

        if not graph_chunks:
            return []

        # Embeddings
        qvec = self.store._embed(question)  # type: ignore[attr-defined]
        doc_vecs = self.store._embed([c.text for c in graph_chunks])  # type: ignore[attr-defined]

        hits: List[RetrievalHit] = []
        for i, (c, v, m) in enumerate(zip(graph_chunks, doc_vecs, meta), start=1):
            sem = _cosine_sim(qvec, v)
            gd = int(m["graph_depth"])
            rels = m["relations"]
            conf = float(m.get("edge_conf", 1.0) or 1.0)

            dd = _depth_decay(gd, gamma=gamma)
            rw = _relation_weight(rels)

            final = float(sem) * float(dd) * float(rw) * float(conf)

            payload = {
                "chunk": c,
                "score": float(final),
                "rank": i,
                "source": "graph",
                "score_breakdown": {
                    "channel": "graph",
                    "semantic": float(sem),
                    "depth_decay": float(dd),
                    "relation_weight": float(rw),
                    "edge_conf": float(conf),
                    "final": float(final),
                    "graph_depth": gd,
                    "relations": rels,
                },
            }
            hits.append(_make_hit(payload))

        # sort by score desc and re-rank
        hits.sort(key=lambda h: float(getattr(h, "score", 0.0) or 0.0), reverse=True)
        for r, h in enumerate(hits, start=1):
            try:
                setattr(h, "rank", r)
            except Exception:
                pass

        return hits[:eff_top_k]
