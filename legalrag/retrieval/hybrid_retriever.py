from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
import traceback
import time
from legalrag.config import AppConfig
from legalrag.schemas import RetrievalHit

from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.retrieval.dense_retriever import DenseRetriever
from legalrag.retrieval.graph_retriever import GraphRetriever
from legalrag.retrieval.colbert_retriever import ColBERTRetriever
from legalrag.retrieval.rerankers import RerankerFactory, rerank_candidates
import torch
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------
# Utils
# -----------------------
def _minmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12: 
        return [0.0 for _ in scores]
    return [(float(s) - lo) / (hi - lo) for s in scores]


def _rrf_with_breakdown(
    rank_lists: Dict[str, List[str]],
    *,
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Returns:
      - rrf_total[cid] = sum over channels w * 1/(k+rank)
      - rrf_contrib_raw[cid][channel] = per-channel raw RRF contribution
    """
    totals: Dict[str, float] = {}
    contrib: Dict[str, Dict[str, float]] = {}
    weights = weights or {}

    for channel, ids in rank_lists.items():
        w = float(weights.get(channel, 1.0))
        for rank, cid in enumerate(ids, start=1):
            v = w * (1.0 / (k + rank))
            totals[cid] = totals.get(cid, 0.0) + v
            contrib.setdefault(cid, {}) 
            contrib[cid][channel] = v

    return totals, contrib


def _as_channel_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, (set, tuple)):
        return [str(i) for i in x]
    if isinstance(x, str):
        return [x]
    return [str(x)]


def _dedup_keep_best(hits: List[RetrievalHit]) -> List[RetrievalHit]:
    """
    Rule:
      - keep the best-scoring hit per chunk.id
      - but UNION provenance:
          * score_breakdown["channel"] is List[str]
          * score_breakdown["channel_contrib"] is Dict[str,float] (summed if present)
    """
    best: Dict[str, RetrievalHit] = {}

    for h in hits:
        cid = h.chunk.id
        sb = h.score_breakdown or {}

        if cid not in best:
            if "channel" in sb:
                sb["channel"] = _as_channel_list(sb.get("channel"))
                h.score_breakdown = sb
            best[cid] = h
            continue

        b = best[cid]
        sb_best = b.score_breakdown or {}

        # union channel list under key "channel"
        chs: Set[str] = set(_as_channel_list(sb_best.get("channel")))
        chs |= set(_as_channel_list(sb.get("channel")))
        merged_channels = list(chs)

        # merge channel_contrib (sum) if present
        merged_contrib: Dict[str, float] = {}
        cb = sb_best.get("channel_contrib", {}) or {}
        cn = sb.get("channel_contrib", {}) or {}
        if isinstance(cb, dict) or isinstance(cn, dict):
            for k, v in (cb or {}).items():
                merged_contrib[str(k)] = merged_contrib.get(str(k), 0.0) + float(v)
            for k, v in (cn or {}).items():
                merged_contrib[str(k)] = merged_contrib.get(str(k), 0.0) + float(v)

        # pick representative (higher score)
        if float(h.score) > float(b.score):
            best[cid] = h

        # write merged provenance back to representative
        rep = best[cid]
        sb_rep = rep.score_breakdown or {}
        if merged_contrib:
            merged_channels.sort(key=lambda c: float(merged_contrib.get(c, 0.0)), reverse=True)
            sb_rep["channel_contrib"] = merged_contrib
        else:
            merged_channels.sort()

        sb_rep["channel"] = merged_channels
        rep.score_breakdown = sb_rep

    out = list(best.values())
    out.sort(key=lambda x: float(x.score), reverse=True)
    for i, h in enumerate(out, start=1):
        h.rank = i
    return out


# -----------------------
# Hybrid Retriever
# -----------------------
@dataclass
class HybridRetriever:
    """
    Channels:
      - dense (FAISS)
      - bm25
      - colbert (optional)
      - graph (optional)

    Fusion methods (cfg.retrieval.fusion_method):
      - "rrf"            : classic RRF
      - "wrrf"           : weighted RRF (per-channel weights)
      - "weighted_sum"   : weighted sum of per-channel minmax-normalized scores
      - "rrf_norm_blend" : alpha * minmax(rrf_total) + (1-alpha) * weighted_sum  (default)

    Notes:
      - score_breakdown["channel"] (singular) is List[str]
      - score_breakdown["channel_contrib"] includes BOTH weighted and RRF contributions (when applicable)
    """

    cfg: AppConfig

    def __post_init__(self) -> None:
        self.dense = DenseRetriever(self.cfg)
        self.bm25 = BM25Retriever(self.cfg)

        self.colbert = None
        if getattr(self.cfg.retrieval, "enable_colbert", False) and ColBERTRetriever is not None:
            try:
                self.colbert = ColBERTRetriever(self.cfg)
            except Exception as e:
                print("[HybridRetriever] ColBERT init failed:", repr(e))
                traceback.print_exc()
                self.colbert = None

        self.graph = None
        if getattr(self.cfg.retrieval, "enable_graph", False) and GraphRetriever is not None:
            try:
                self.graph = GraphRetriever(self.cfg)
            except Exception:
                self.graph = None

    # -----------------------
    # Per-channel APIs  
    # -----------------------
    def search_dense(self, question: str, top_k: int = 10) -> List[RetrievalHit]:
        top_k = max(1, int(top_k))
        hits = self.dense.search(question, top_k)
        hits.sort(key=lambda h: float(h.score), reverse=True)
        for i, h in enumerate(hits, start=1):
            h.rank = i
            h.source = "retriever"
            h.score_breakdown = {"channel": ["dense"], "dense_raw": float(h.score)}
        return hits

    def search_bm25(self, question: str, top_k: int = 10) -> List[RetrievalHit]:
        top_k = max(1, int(top_k))
        pairs = self.bm25.search(question, top_k)
        hits: List[RetrievalHit] = []
        for i, (c, s) in enumerate(pairs, start=1):
            hits.append(
                RetrievalHit(
                    chunk=c,
                    score=float(s),
                    rank=i,
                    source="retriever",
                    score_breakdown={"channel": ["bm25"], "bm25_raw": float(s)},
                )
            )
        hits.sort(key=lambda h: float(h.score), reverse=True)
        for i, h in enumerate(hits, start=1):
            h.rank = i
        return hits

    def search_colbert(self, question: str, top_k: int = 10) -> List[RetrievalHit]:
        top_k = max(1, int(top_k))
        if self.colbert is None:
            return []
        try:
            raw = self.colbert.search(question, top_k)
            hits: List[RetrievalHit] = []

            # ColBERT retriever may return either RetrievalHit or (chunk, score)
            for item in raw:
                if isinstance(item, RetrievalHit):
                    hits.append(item)
                else:
                    c, s = item   
                    hits.append(
                        RetrievalHit(
                            chunk=c,
                            score=float(s),
                            rank=0,
                            source="retriever",
                            score_breakdown={"channel": ["colbert"], "colbert_raw": float(s)},
                        )
                    )

            hits.sort(key=lambda h: float(h.score), reverse=True)
            for i, h in enumerate(hits, start=1):
                h.rank = i
                h.source = "retriever"
                sb = h.score_breakdown or {}
                sb["channel"] = _as_channel_list(sb.get("channel")) or ["colbert"]
                if "colbert_raw" not in sb:
                    sb["colbert_raw"] = float(h.score)
                h.score_breakdown = sb
            return hits
        except Exception:
            return []

    def search_graph(
        self,
        question: str,
        top_k: int = 10,
        *,
        decision: Any = None,
        seeds: Optional[List[RetrievalHit]] = None,
    ) -> List[RetrievalHit]:
        top_k = max(1, int(top_k))
        if self.graph is None:
            return []

        if seeds is None:
            seed_n = int(getattr(self.cfg.retrieval, "graph_seed_k", max(10, top_k * 3)))
            dense = self.search_dense(question, seed_n)
            bm25 = self.search_bm25(question, seed_n)
            colb = self.search_colbert(question, seed_n)
            seeds = dense[:seed_n] + bm25[:seed_n] + colb[:seed_n]

        try:
            hits = self.graph.search(question, seeds, decision=decision, top_k=top_k)
            hits.sort(key=lambda h: float(h.score), reverse=True)
            for i, h in enumerate(hits, start=1):
                h.rank = i
                h.source = "retriever"
                sb = h.score_breakdown or {}
                sb["channel"] = _as_channel_list(sb.get("channel")) or ["graph"]
                h.score_breakdown = sb
            return hits
        except Exception:
            return []

    # -----------------------
    # Main search
    # -----------------------
    def search(self, question: str, llm: Optional[LLMClient] = None, top_k: int = 10, decision: Any = None) -> List[RetrievalHit]:
        rcfg = self.cfg.retrieval
        top_k = max(1, int(top_k))

        has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        t_start = time.time()

        eff_top_k = int(getattr(rcfg, "top_k", top_k * 8) or (top_k * 8))
        if eff_top_k < top_k:
            eff_top_k = top_k

        # retrieve per channel (oversampled)
        t0 = time.time()
        dense_hits = self.search_dense(question, eff_top_k)
        t1 = time.time()
        bm25_hits = self.search_bm25(question, eff_top_k)
        t2 = time.time()
        colbert_hits = self.search_colbert(question, eff_top_k)
        t3 = time.time()

        fused = self._fuse(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            colbert_hits=colbert_hits 
        )
        t4 = time.time()

        min_final = float(getattr(rcfg, "min_final_score", 0.0))
        fused = [h for h in fused if float(h.score) >= min_final]

        graph_hits: List[RetrievalHit] = []
        t_graph = None
        mode = getattr(decision, "mode", None)
        if getattr(rcfg, "enable_graph", False) and mode and ( (str(mode).upper().endswith("GRAPH_AUGMENTED") or str(mode) == "RoutingMode.GRAPH_AUGMENTED")):
            seed_n = int(getattr(rcfg, "graph_seed_k", max(10, top_k * 3)))
            seeds = fused[:seed_n]
            graph_hits = self.search_graph(question, eff_top_k, decision=decision, seeds=seeds)
            fused = seeds + graph_hits
            t_graph = time.time()

        # optional rerank
        t_rerank = None
        if getattr(rcfg, "enable_rerank", False):

            # 1) CrossEncoder / LLM 
            use_llm_rerank = bool(getattr(rcfg, "rerank_use_llm", False))
            factory = RerankerFactory( 
                llm=llm if use_llm_rerank else None, 
                cross_model=rcfg.rerank_ce_model,
                llm_threshold=30, 
                use_cache=True, 
            )

            rerank_top_n = int(getattr(rcfg, "rerank_top_n", min(40, max(10, top_k * 4))))
            beta = float(getattr(rcfg, "rerank_beta", 0.35))

            cand = fused[:rerank_top_n]
            docs = [h.chunk.text for h in cand] 

            if docs:
                reranker = factory.create(top_k=len(cand))
                reranked = rerank_candidates(query=question, candidates=cand, reranker=reranker, top_n=len(cand), normalize="minmax", include_debug=True)
                new_hits = [] 
                for (hit, rr) in reranked: 
                    hit.score_breakdown = hit.score_breakdown or {} 
                    hit.score_breakdown.update({ "rerank_raw": rr.raw_score, "rerank_norm": rr.norm_score, "rerank_beta": beta, }) 
                    hit.score = (1 - beta) * float(hit.score) + beta * float(rr.norm_score) 
                    hit.source = "rerank" 
                    new_hits.append(hit)

                fused[:len(new_hits)] = new_hits
                fused.sort(key=lambda x: float(x.score), reverse=True)
                for i, h in enumerate(fused, start=1):
                    h.rank = i
            t_rerank = time.time()

        # defensive dedup  
        fused = _dedup_keep_best(fused)
        t_end = time.time()
        ms = lambda a, b: int((b - a) * 1000)
        dense_ms = ms(t0, t1)
        bm25_ms = ms(t1, t2)
        colbert_ms = ms(t2, t3)
        fuse_ms = ms(t3, t4)
        graph_ms = ms(t4, t_graph) if t_graph else 0
        rerank_ms = ms((t_graph or t4), t_rerank) if t_rerank else 0
        total_ms = ms(t_start, t_end)
        graph_enabled = bool(getattr(rcfg, "enable_graph", False))
        colbert_enabled = self.colbert is not None
        logger.info(
            "[retrieval] dense=%dms bm25=%dms colbert=%dms fuse=%dms graph=%dms rerank=%dms total=%dms enabled(graph=%s,colbert=%s, has_gpu=%s)",
            dense_ms,
            bm25_ms,
            colbert_ms,
            fuse_ms,
            graph_ms,
            rerank_ms,
            total_ms,
            int(graph_enabled),
            int(colbert_enabled), 
            int(has_gpu),
        )
        return fused[:top_k]

    # -----------------------
    # Fusion
    # -----------------------
    def _fuse(
        self,
        *,
        dense_hits: List[RetrievalHit],
        bm25_hits: List[RetrievalHit],
        colbert_hits: List[RetrievalHit], 
    ) -> List[RetrievalHit]:
        rcfg = self.cfg.retrieval

        method = str(getattr(rcfg, "fusion_method", "rrf_norm_blend")).lower()
        rrf_k = int(getattr(rcfg, "rrf_k", 60))
        alpha = float(getattr(rcfg, "rrf_alpha", 0.50))  # used for rrf_norm_blend

        weights: Dict[str, float] = {
            "dense": float(getattr(rcfg, "dense_weight", 0.55)),
            "bm25": float(getattr(rcfg, "bm25_weight", 0.35)),
            "colbert": float(getattr(rcfg, "colbert_weight", 0.25)) 
        }

        # ensure sorted by score desc  
        dense_hits.sort(key=lambda h: float(h.score), reverse=True)
        bm25_hits.sort(key=lambda h: float(h.score), reverse=True)
        colbert_hits.sort(key=lambda h: float(h.score), reverse=True) 

        # rank lists
        rank_lists: Dict[str, List[str]] = {
            "dense": [h.chunk.id for h in dense_hits],
            "bm25": [h.chunk.id for h in bm25_hits],
            "colbert": [h.chunk.id for h in colbert_hits] 
        }

        # membership (retrieved channels)
        channels_by_id: Dict[str, Set[str]] = {}
        for ch, ids in rank_lists.items():
            for cid in ids:
                channels_by_id.setdefault(cid, set()).add(ch)

        # chunk lookup (prefer dense -> bm25 -> colbert -> graph) via setdefault order
        chunk_by_id: Dict[str, Any] = {}
        for h in dense_hits + bm25_hits + colbert_hits:
            chunk_by_id.setdefault(h.chunk.id, h.chunk)

        # per-channel minmax normalized maps
        def _norm_map(hits: List[RetrievalHit]) -> Dict[str, float]:
            vals = _minmax([float(h.score) for h in hits])
            out: Dict[str, float] = {}
            for i, h in enumerate(hits):
                out[h.chunk.id] = float(vals[i]) if i < len(vals) else 0.0
            return out

        norm_map_by_channel: Dict[str, Dict[str, float]] = {
            "dense": _norm_map(dense_hits),
            "bm25": _norm_map(bm25_hits),
            "colbert": _norm_map(colbert_hits) 
        }

        # RRF totals and raw per-channel contribution
        if method == "wrrf":
            rrf_total, rrf_contrib_raw = _rrf_with_breakdown(rank_lists, k=rrf_k, weights=weights)
        else:
            rrf_total, rrf_contrib_raw = _rrf_with_breakdown(rank_lists, k=rrf_k)

        # minmax normalize rrf_total
        rrf_norm_map: Dict[str, float] = {}
        if rrf_total:
            items = list(rrf_total.items())
            vals = _minmax([float(v) for _, v in items])
            for i, (cid, _) in enumerate(items):
                rrf_norm_map[cid] = float(vals[i]) if i < len(vals) else 0.0

        # candidates
        all_ids: Set[str] = set(rrf_total)
        for m in norm_map_by_channel.values():
            all_ids |= set(m)

        # helpers for per-id computation 
        def _get_norms(cid: str) -> Dict[str, float]:
            return {ch: float(norm_map_by_channel[ch].get(cid, 0.0)) for ch in weights}

        def _weighted_terms(norms: Dict[str, float]) -> Dict[str, float]:
            return {ch: float(weights[ch]) * float(norms[ch]) for ch in weights}

        def _rrf_alloc(cid: str, mass: float) -> Dict[str, float]:
            raw = rrf_contrib_raw.get(cid, {}) or {}
            total = float(rrf_total.get(cid, 0.0))
            if mass <= 0.0 or total <= 1e-18:
                return {}
            return {str(ch): mass * float(v) / total for ch, v in raw.items()}

        # compute fused score and per channel contrib 
        score_cache: Dict[str, float] = {}
        contrib_cache: Dict[str, Dict[str, float]] = {}
        channel_list_cache: Dict[str, List[str]] = {}
        weighted_sum_map: Dict[str, float] = {}

        for cid in all_ids:
            norms = _get_norms(cid)
            w_terms = _weighted_terms(norms)
            wsum = sum(w_terms.values())
            weighted_sum_map[cid] = float(wsum)
            rrf_norm = float(rrf_norm_map.get(cid, 0.0))
            
            contrib: Dict[str, float] = {ch: 0.0 for ch in weights}

            if method == "weighted_sum":
                score = float(wsum)
                contrib.update(w_terms)

            elif method in ("rrf", "wrrf"):
                score = float(rrf_norm)
                contrib.update(_rrf_alloc(cid, score))

            else:
                # rrf_norm_blend (default)
                score = float(alpha) * float(rrf_norm) + (1.0 - float(alpha)) * float(wsum)

                # weighted part
                for ch, v in w_terms.items():
                    contrib[ch] += (1.0 - float(alpha)) * float(v)

                # RRF part
                for ch, v in _rrf_alloc(cid, float(alpha) * float(rrf_norm)).items():
                    contrib[ch] = contrib.get(ch, 0.0) + float(v)

            score_cache[cid] = float(score)
            contrib_cache[cid] = contrib

            # channel list: membership U rrf contributors; sort by contribution desc  
            membership = set(channels_by_id.get(cid, set()) or []) 
            ch_list = sorted(
                list(membership),
                key=lambda c: (float(contrib.get(str(c), 0.0)), str(c)),
                reverse=True,
            )
            channel_list_cache[cid] = ch_list

        # build output list sorted by fused score
        ranked = sorted(score_cache.items(), key=lambda x: float(x[1]), reverse=True)
        out: List[RetrievalHit] = []

        for r, (cid, score) in enumerate(ranked, start=1):
            chunk = chunk_by_id.get(cid)
            if chunk is None:
                continue

            sb = {
                "fusion_method": method,
                "rrf_k": int(rrf_k),
                "alpha": float(alpha),
                "channel_weights": dict(weights),
                "channel": channel_list_cache.get(cid, []),
                "channel_contrib": contrib_cache.get(cid, {}),
                # debug components
                "rrf_norm": float(rrf_norm_map.get(cid, 0.0)),
                "weighted_sum": float(weighted_sum_map.get(cid, 0.0)),
                "dense_norm": float(norm_map_by_channel["dense"].get(cid, 0.0)),
                "bm25_norm": float(norm_map_by_channel["bm25"].get(cid, 0.0)),
                "colbert_norm": float(norm_map_by_channel["colbert"].get(cid, 0.0))
            }

            out.append(RetrievalHit(chunk=chunk, score=float(score), rank=r, source="retriever", score_breakdown=sb))

        return out
