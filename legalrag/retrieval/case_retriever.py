from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Optional, Tuple
import json

import jieba
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi

from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.retrieval.rerankers import CrossEncoderReranker, LLMReranker, minmax_normalize
from legalrag.schemas import CaseEntry, CaseRetrievalHit
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def _case_text(case: CaseEntry) -> str:
    parts = [case.title, case.facts, case.issue, case.holding]
    return "\n".join([p for p in parts if p])


@dataclass
class CaseVectorStore:
    cfg: AppConfig
    index_path: Path
    meta_path: Path
    model_name: str
    _instances_by_key: ClassVar[Dict[Tuple[str, str, str, str], "CaseVectorStore"]] = {}

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        *,
        index_path: Path,
        meta_path: Path,
        model_name: str,
    ) -> "CaseVectorStore":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = (str(index_path), str(meta_path), str(model_name), device)
        if key in cls._instances_by_key:
            return cls._instances_by_key[key]
        inst = cls(cfg=cfg, index_path=index_path, meta_path=meta_path, model_name=model_name)
        cls._instances_by_key[key] = inst
        return inst

    def __post_init__(self) -> None:
        from FlagEmbedding import FlagModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlagModel(
            self.model_name,
            query_instruction_for_retrieval="为这个问题生成表示以用于检索相关案例：",
            use_fp16=torch.cuda.is_available(),
            device=self.device,
        )
        self.index: faiss.Index | None = None
        self.cases: List[CaseEntry] = []

    def _embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            dim = self.model.model.config.hidden_size
            return np.zeros((0, dim), dtype="float32")
        if is_query:
            embs = self.model.encode_queries(texts, batch_size=64, max_length=512)
        else:
            embs = self.model.encode(texts, batch_size=64, max_length=512)
        return embs.astype("float32")

    def load(self) -> None:
        if self.index is not None and self.cases:
            return
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Case FAISS index or meta not found; build case index first.")
        self.index = faiss.read_index(str(self.index_path))
        self.cases = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.cases.append(CaseEntry.model_validate(json.loads(line)))

    def build(self, cases: List[CaseEntry]) -> None:
        self.cases = list(cases)
        texts = [_case_text(c) for c in self.cases]
        embs = self._embed(texts, is_query=False)
        dim = embs.shape[1] if embs.size else 768
        index = faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
        if embs.size:
            index.add(embs)
        self.index = index
        self.persist()

    def persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for c in self.cases:
                f.write(c.model_dump_json() + "\n")


class CaseRetriever:
    def __init__(self, cfg: AppConfig, path: str | None = None) -> None:
        self.cfg = cfg
        rcfg = cfg.retrieval
        has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.path = Path(path or rcfg.case_file)
        self.cases: List[CaseEntry] = []
        self.bm25: BM25Okapi | None = None
        self.vector_store = CaseVectorStore.from_config(
            cfg,
            index_path=Path(rcfg.case_index_file),
            meta_path=Path(rcfg.case_meta_file),
            model_name=rcfg.case_embedding_model,
        )
        self.reranker = None
        if has_gpu and rcfg.case_enable_rerank and rcfg.case_rerank_model:
            self.reranker = CrossEncoderReranker(model_name=rcfg.case_rerank_model)
        self.llm_reranker = None
        if has_gpu and rcfg.case_use_llm_rerank:
            self.llm_reranker = LLMReranker(LLMClient.from_config(cfg))
        if not has_gpu and (rcfg.case_enable_rerank or rcfg.case_use_llm_rerank):
            logger.info("[CaseRetriever] no GPU detected; skip rerankers")
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            logger.warning("[CaseRetriever] case file not found: %s", self.path)
            self.cases = []
            self.bm25 = None
            return
        cases: List[CaseEntry] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cases.append(CaseEntry(**obj))
        self.cases = cases
        corpus = [_case_text(c) for c in cases]
        tokenized = [tokenize(t) for t in corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        try:
            self.vector_store.load()
        except Exception:
            # build index if missing
            if self.cases:
                self.vector_store.build(self.cases)
        logger.info("[CaseRetriever] loaded %d cases from %s", len(self.cases), self.path)

    def _filter_cases(self, cases: Iterable[CaseEntry], filters: Optional[Dict[str, str]] = None) -> List[CaseEntry]:
        if not filters:
            return list(cases)
        out: List[CaseEntry] = []
        for c in cases:
            ok = True
            for k, v in filters.items():
                cv = getattr(c, k, None)
                if cv is None:
                    ok = False
                    break
                if isinstance(cv, list):
                    if v not in cv:
                        ok = False
                        break
                else:
                    if str(v) not in str(cv):
                        ok = False
                        break
            if ok:
                out.append(c)
        return out

    def _bm25_hits(self, query: str, top_k: int) -> List[CaseRetrievalHit]:
        if not self.cases or self.bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits: List[CaseRetrievalHit] = []
        for rank, i in enumerate(ranked, start=1):
            if scores[i] <= 0:
                continue
            hits.append(CaseRetrievalHit(case=self.cases[i], score=float(scores[i]), rank=rank, source="bm25"))
        return hits

    def _dense_hits(self, query: str, top_k: int) -> List[CaseRetrievalHit]:
        try:
            self.vector_store.load()
        except Exception:
            return []
        q_vec = self.vector_store._embed([query], is_query=True)
        scores, idxs = self.vector_store.index.search(q_vec, max(1, int(top_k)))
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        hits: List[CaseRetrievalHit] = []
        for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
            if i < 0 or i >= len(self.vector_store.cases):
                continue
            hits.append(
                CaseRetrievalHit(case=self.vector_store.cases[i], score=float(s), rank=rank, source="dense")
            )
        return hits

    def _rrf(self, bm25_hits: List[CaseRetrievalHit], dense_hits: List[CaseRetrievalHit]) -> List[CaseRetrievalHit]:
        rrf_k = int(getattr(self.cfg.retrieval, "case_rrf_k", 60))
        scores: Dict[str, float] = {}
        case_map: Dict[str, CaseEntry] = {}

        def _accumulate(hits: List[CaseRetrievalHit], weight: float) -> None:
            for h in hits:
                cid = h.case.id
                case_map[cid] = h.case
                scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + (h.rank or 1))

        _accumulate(bm25_hits, 1.0)
        _accumulate(dense_hits, 1.0)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            CaseRetrievalHit(case=case_map[cid], score=float(score), rank=i + 1, source="rrf")
            for i, (cid, score) in enumerate(ranked)
        ]

    def _rerank(self, query: str, hits: List[CaseRetrievalHit], top_k: int) -> List[CaseRetrievalHit]:
        if not hits:
            return []
        docs = [_case_text(h.case) for h in hits]
        if self.llm_reranker is not None:
            scores = self.llm_reranker.score_batch(query, docs)
        elif self.reranker is not None:
            scores = self.reranker.score_batch(query, docs)
        else:
            return hits[:top_k]
        norm = minmax_normalize(scores)
        ranked = sorted(
            zip(hits, norm), key=lambda x: x[1], reverse=True
        )[:top_k]
        out: List[CaseRetrievalHit] = []
        for rank, (h, s) in enumerate(ranked, start=1):
            out.append(
                CaseRetrievalHit(case=h.case, score=float(s), rank=rank, source="rerank")
            )
        return out

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[CaseRetrievalHit]:
        if not self.cases:
            return []
        k = int(top_k or self.cfg.retrieval.case_top_k)

        bm25_hits: List[CaseRetrievalHit] = []
        dense_hits: List[CaseRetrievalHit] = []
        if self.cfg.retrieval.case_enable_bm25:
            bm25_hits = self._bm25_hits(query, max(k, 10))
        if self.cfg.retrieval.case_enable_dense:
            dense_hits = self._dense_hits(query, max(k, 10))

        fused = self._rrf(bm25_hits, dense_hits) if (bm25_hits or dense_hits) else []
        fused = fused[: max(k * 3, k)]

        if filters:
            allowed = {c.id for c in self._filter_cases([h.case for h in fused], filters)}
            fused = [h for h in fused if h.case.id in allowed]

        if self.cfg.retrieval.case_enable_rerank or self.cfg.retrieval.case_use_llm_rerank:
            return self._rerank(query, fused, k)
        return fused[:k]

    def add_cases(self, new_cases: List[CaseEntry], rebuild_index: bool = True) -> None:
        if not new_cases:
            return
        self.cases.extend(new_cases)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for c in new_cases:
                f.write(c.model_dump_json() + "\n")
        if rebuild_index:
            self._load()

    def add_case(self, case: CaseEntry, rebuild_index: bool = True) -> None:
        self.add_cases([case], rebuild_index=rebuild_index)
