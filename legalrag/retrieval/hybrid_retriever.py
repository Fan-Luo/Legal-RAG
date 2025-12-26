from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from legalrag.config import AppConfig
from legalrag.schemas import RetrievalHit
from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.retrieval.dense_retriever import DenseRetriever
from legalrag.retrieval.graph_retriever import GraphRetriever
from legalrag.retrieval.colbert_retriever import ColBERTRetriever   

from legalrag.retrieval.rerankers import CrossEncoderReranker, LLMReranker   


def _minmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(float(s) - lo) / (hi - lo) for s in scores]


def _rrf(rank_lists: Dict[str, List[str]], *, k: int = 60, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    weights = weights or {}
    for ch, ids in rank_lists.items():
        w = float(weights.get(ch, 1.0))
        for r, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w * (1.0 / (k + r))
    return scores


def _dedup_keep_best(hits: List[RetrievalHit]) -> List[RetrievalHit]:
    best: Dict[str, RetrievalHit] = {}
    for h in hits:
        cid = h.chunk.id
        if cid not in best or float(h.score) > float(best[cid].score):
            best[cid] = h
    out = list(best.values())
    out.sort(key=lambda x: float(x.score), reverse=True)
    for i, h in enumerate(out, start=1):
        h.rank = i
    return out


@dataclass
class HybridRetriever:
    """ 
    Channels:
      - dense (FAISS)
      - bm25
      - colbert (optional)
      - graph (optional; enabled only under RoutingMode.GRAPH_AUGMENTED)

    Fusion methods (cfg.retrieval.fusion_method):
      - "rrf"           : classic RRF
      - "wrrf"          : weighted RRF (per-channel)
      - "weighted_sum"  : weighted sum of per-channel minmax-normalized scores
      - "rrf_norm_blend": alpha * minmax(rrf) + (1-alpha) * weighted_sum  (default)

    Rerank:
      - CrossEncoder (recommended) or LLM reranker (optional)
      - rerank_top_n candidates from fused pool
      - blend back by beta
    """

    cfg: AppConfig

    def __post_init__(self) -> None:
        self.dense = DenseRetriever(self.cfg)
        self.bm25 = BM25Retriever(self.cfg)

        self.colbert = None
        if getattr(self.cfg.retrieval, "enable_colbert", False) and ColBERTRetriever is not None:
            try:
                self.colbert = ColBERTRetriever(self.cfg)  # type: ignore
            except Exception:
                self.colbert = None

        self.graph = GraphRetriever(self.cfg)

        self.reranker = None
        if getattr(self.cfg.retrieval, "enable_rerank", False):
            mode = str(getattr(self.cfg.retrieval, "rerank_mode", "cross_encoder")).lower()
            if mode in ("cross_encoder", "ce") and CrossEncoderReranker is not None:
                try:
                    self.reranker = CrossEncoderReranker(model_name=self.cfg.retrieval.rerank_model)
                except Exception:
                    self.reranker = None
            elif mode in ("llm", "openai") and LLMReranker is not None:
                try:
                    self.reranker = LLMReranker(self.cfg)  
                except Exception:
                    self.reranker = None

    def search(self, question: str, top_k: int = 10, decision: Any = None) -> List[RetrievalHit]:
        rcfg = self.cfg.retrieval
        eff_top_k = max(1, int(top_k))

        # 1) Candidate generation (oversample)
        dense_k = int(getattr(rcfg, "dense_candidate_k", eff_top_k * 8))
        bm25_k = int(getattr(rcfg, "bm25_candidate_k", eff_top_k * 8))
        colbert_k = int(getattr(rcfg, "colbert_candidate_k", eff_top_k * 8))
        graph_k = int(getattr(rcfg, "graph_candidate_k", eff_top_k * 8))

        dense_hits = self.dense.search(question, dense_k)
        for h in dense_hits:
            h.source = "retriever"
            h.score_breakdown = {"channel": "dense", "dense_raw": float(h.score)}

        bm25_pairs = self.bm25.search(question, bm25_k)
        bm25_hits: List[RetrievalHit] = []
        for i, (c, s) in enumerate(bm25_pairs, start=1):
            bm25_hits.append(
                RetrievalHit(
                    chunk=c,
                    score=float(s),
                    rank=i,
                    source="retriever",
                    score_breakdown={"channel": "bm25", "bm25_raw": float(s)},
                )
            )

        colbert_hits: List[RetrievalHit] = []
        if self.colbert is not None:
            try:
                colbert_hits = self.colbert.search(question, colbert_k)  # type: ignore
                for h in colbert_hits:
                    h.source = "retriever"
                    h.score_breakdown = {"channel": "colbert", "colbert_raw": float(h.score)}
            except Exception:
                colbert_hits = []

        # 2) Graph channel (independent)
        graph_hits: List[RetrievalHit] = []
        mode = getattr(decision, "mode", None)
        graph_enabled = str(mode).upper().endswith("GRAPH_AUGMENTED") or str(mode) == "RoutingMode.GRAPH_AUGMENTED"
        if graph_enabled:
            seed_n = int(getattr(rcfg, "graph_seed_k", max(10, eff_top_k * 3)))
            seeds = dense_hits[:seed_n] + bm25_hits[:seed_n] + colbert_hits[:seed_n]
            graph_hits = self.graph.search(question, seeds, decision=decision, top_k=graph_k)

        # 3) Fusion
        fused = self._fuse(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            colbert_hits=colbert_hits,
            graph_hits=graph_hits,
            pool_k=int(getattr(rcfg, "fusion_pool_k", eff_top_k * 12)),
        )

        min_final = float(getattr(rcfg, "min_final_score", 0.0))
        fused = [h for h in fused if float(h.score) >= min_final]

        # 4) Optional rerank
        if self.reranker is not None and fused:
            top_n = int(getattr(rcfg, "rerank_top_n", 50))
            beta = float(getattr(rcfg, "rerank_blend_beta", 0.35))

            cand = fused[: max(1, min(top_n, len(fused)))]
            docs = [h.chunk.text for h in cand]

            # Reranker protocol: score_batch preferred; fallback to score
            if hasattr(self.reranker, "score_batch"):
                r_raw = self.reranker.score_batch(question, docs)  # type: ignore[attr-defined]
            else:
                r_raw = [float(self.reranker.score(question, d)) for d in docs]  # type: ignore[attr-defined]

            r_norm = _minmax(r_raw)

            for i, h in enumerate(cand):
                h.score_breakdown = h.score_breakdown or {}
                h.score_breakdown.update(
                    {
                        "rerank_raw": float(r_raw[i]),
                        "rerank_norm": float(r_norm[i]),
                        "rerank_beta": float(beta),
                    }
                )
                h.score = (1.0 - beta) * float(h.score) + beta * float(r_norm[i])
                h.source = "rerank"

            fused[: len(cand)] = cand
            fused.sort(key=lambda x: float(x.score), reverse=True)
            for i, h in enumerate(fused, start=1):
                h.rank = i

        fused = _dedup_keep_best(fused)
        return fused[:eff_top_k]

    def _fuse(
        self,
        *,
        dense_hits: List[RetrievalHit],
        bm25_hits: List[RetrievalHit],
        colbert_hits: List[RetrievalHit],
        graph_hits: List[RetrievalHit],
        pool_k: int,
    ) -> List[RetrievalHit]:
        rcfg = self.cfg.retrieval
        method = str(getattr(rcfg, "fusion_method", "rrf_norm_blend")).lower()
        rrf_k = int(getattr(rcfg, "rrf_k", 60))
        alpha = float(getattr(rcfg, "rrf_blend_alpha", 0.6))

        # Per-channel normalization
        dense_norm = _minmax([h.score for h in dense_hits])
        bm25_norm = _minmax([h.score for h in bm25_hits])
        colbert_norm = _minmax([h.score for h in colbert_hits])
        graph_norm = _minmax([h.score for h in graph_hits])

        dense_map = {h.chunk.id: dense_norm[i] for i, h in enumerate(dense_hits)}
        bm25_map = {h.chunk.id: bm25_norm[i] for i, h in enumerate(bm25_hits)}
        colbert_map = {h.chunk.id: colbert_norm[i] for i, h in enumerate(colbert_hits)}
        graph_map = {h.chunk.id: graph_norm[i] for i, h in enumerate(graph_hits)}

        # Rank lists for RRF
        rank_lists: Dict[str, List[str]] = {
            "dense": [h.chunk.id for h in dense_hits],
            "bm25": [h.chunk.id for h in bm25_hits],
            "colbert": [h.chunk.id for h in colbert_hits],
            "graph": [h.chunk.id for h in graph_hits],
        }

        # weights
        w_dense = float(getattr(rcfg, "dense_weight", 0.55))
        w_bm25 = float(getattr(rcfg, "bm25_weight", 0.35))
        w_colbert = float(getattr(rcfg, "colbert_weight", 0.25))
        w_graph = float(getattr(rcfg, "graph_weight", 0.20))

        # RRF scores
        if method == "wrrf":
            rrf_scores = _rrf(
                rank_lists,
                k=rrf_k,
                weights={"dense": w_dense, "bm25": w_bm25, "colbert": w_colbert, "graph": w_graph},
            )
        else:
            rrf_scores = _rrf(rank_lists, k=rrf_k)

        # normalize rrf for blend
        rrf_norm_scores = _minmax(list(rrf_scores.values()))
        rrf_norm_map = {cid: rrf_norm_scores[i] for i, cid in enumerate(rrf_scores.keys())}

        # weighted sum over normalized channel scores
        all_ids = set(dense_map) | set(bm25_map) | set(colbert_map) | set(graph_map) | set(rrf_scores)
        weighted_map: Dict[str, float] = {}
        for cid in all_ids:
            weighted_map[cid] = (
                w_dense * dense_map.get(cid, 0.0)
                + w_bm25 * bm25_map.get(cid, 0.0)
                + w_colbert * colbert_map.get(cid, 0.0)
                + w_graph * graph_map.get(cid, 0.0)
            )

        fused_map: Dict[str, float] = {}
        if method in ("rrf", "wrrf"):
            fused_map = {cid: float(rrf_scores.get(cid, 0.0)) for cid in all_ids}
        elif method == "weighted_sum":
            fused_map = dict(weighted_map)
        else:
            # default: rrf_norm_blend
            for cid in all_ids:
                fused_map[cid] = alpha * rrf_norm_map.get(cid, 0.0) + (1.0 - alpha) * weighted_map.get(cid, 0.0)

        # Build chunk lookup
        chunk_by_id: Dict[str, Any] = {}
        for h in dense_hits + bm25_hits + colbert_hits + graph_hits:
            chunk_by_id[h.chunk.id] = h.chunk

        items = sorted(fused_map.items(), key=lambda x: float(x[1]), reverse=True)[: max(1, int(pool_k))]
        out: List[RetrievalHit] = []
        for r, (cid, s) in enumerate(items, start=1):
            ch = chunk_by_id.get(cid)
            if ch is None:
                continue
            sb = {
                "fusion_method": method,
                "rrf_k": int(rrf_k),
                "rrf_norm": float(rrf_norm_map.get(cid, 0.0)),
                "weighted_norm": float(weighted_map.get(cid, 0.0)),
                "dense_norm": float(dense_map.get(cid, 0.0)),
                "bm25_norm": float(bm25_map.get(cid, 0.0)),
                "colbert_norm": float(colbert_map.get(cid, 0.0)),
                "graph_norm": float(graph_map.get(cid, 0.0)),
                "alpha": float(alpha),
                "channel_weights": {"dense": w_dense, "bm25": w_bm25, "colbert": w_colbert, "graph": w_graph},
            }
            out.append(RetrievalHit(chunk=ch, score=float(s), rank=r, source="retriever", score_breakdown=sb))
        return out
