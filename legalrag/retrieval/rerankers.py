from __future__ import annotations

""" 
Works with CrossEncoder and LLM-based rerankers.
"""

from dataclasses import dataclass
from pydantic import BaseModel, PrivateAttr
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union
import math
import re
import json
import asyncio
from legalrag.llm.client import LLMClient


TextLike = Union[str, Dict[str, Any]]


class BaseReranker(Protocol):
    """Synchronous reranker interface."""

    def score(self, query: str, doc: str) -> float: ...

    def score_batch(self, query: str, docs: List[str]) -> List[float]:
        return [self.score(query, d) for d in docs]


class AsyncReranker(Protocol):
    """Async reranker interface."""

    async def score(self, query: str, doc: str) -> float: ...

    async def score_batch(self, query: str, docs: List[str]) -> List[float]: ...


@dataclass(frozen=True)
class RerankResult:
    raw_score: float
    norm_score: float
    meta: Dict[str, Any]


# ---------------------------
# Normalization
# ---------------------------

def minmax_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_calibrate(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    t = max(1e-6, float(temperature))
    return [sigmoid(s / t) for s in scores]


def _safe_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------
# Helpers
# ---------------------------

def _to_doc_text(doc: TextLike, content_key: str = "text") -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for k in (content_key, "text", "content", "provision", "chunk", "body"):
            if k in doc and isinstance(doc[k], str):
                return doc[k]
        return str(doc)
    return str(doc)


# ---------------------------
# CrossEncoder reranker
# ---------------------------

@dataclass
class CrossEncoderReranker:
    model_name: str = "BAAI/bge-reranker-base"
    device: Optional[str] = None
    max_length: int = 512
    batch_size: int = 32

    def __post_init__(self):
        from sentence_transformers import CrossEncoder
        import torch

        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def score(self, query: str, doc: str) -> float:
        return float(self._model.predict([(query, doc)])[0])

    def score_batch(self, query: str, docs: List[str]) -> List[float]:
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


# ---------------------------
# LLM reranker (sync)
# ---------------------------

LLM_RERANK_SYSTEM_PROMPT = """
You are a precise ranking model. 
Your task is to evaluate how well a candidate legal provisions answers a user query.

Rules:
- Output ONLY a JSON object.
- JSON format: {"score": float, "reason": "string"}
- score MUST be between 0 and 1.
- score=1.0 means the provision directly answers the query.
- score=0.0 means the provision is irrelevant.
- Keep "reason" short.
- Do NOT add any text outside JSON.
"""

def build_llm_rerank_prompt(query: str, provision: str) -> str:
    return f"""
Query:
{query}

Candidate provision:
{provision}

Evaluate relevance and return JSON only.
"""


@dataclass
class LLMReranker:
    llm: LLMClient
    temperature: float = 0.0

    max_query_chars: int = 800
    max_doc_chars: int = 2000

    def _truncate(self, s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + "…"

    def _call_llm(self, query: str, doc: str) -> str:
        q = self._truncate(query, self.max_query_chars)
        d = self._truncate(doc, self.max_doc_chars)

        messages = [
            {"role": "system", "content": LLM_RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_rerank_prompt(q, d)},
        ]
        return str(self.llm.chat(messages=messages, tag="rerank_llm"))

    def score(self, query: str, doc: str) -> float:
        text = self._call_llm(query, doc)
        return _safe_clip(self._extract_score(text), 0.0, 1.0)

    def score_batch(self, query: str, docs: List[str]) -> List[float]:
        return [self.score(query, d) for d in docs]

    @staticmethod
    def _extract_score(text: str) -> float:
        t = (text or "").strip()

        # JSON first
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception:
            pass

        # fallback: any float 0..1
        m = re.search(r"([0-1](?:\.\d+)?)", t)
        if m:
            return float(m.group(1))

        return 0.0


# ---------------------------
# Async LLM reranker
# ---------------------------

@dataclass
class AsyncLLMReranker:
    llm: LLMClient
    max_concurrency: int = 8

    base: Optional[LLMReranker] = None

    def __post_init__(self):
        if self.base is None:
            self.base = LLMReranker(llm=self.llm)

    async def _call_llm_async(self, query: str, doc: str) -> str:
        if hasattr(self.llm, "achat"):
            messages = [
                {"role": "system", "content": LLM_RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": build_llm_rerank_prompt(query, doc)},
            ]
            return str(await self.llm.achat(messages=messages, tag="rerank_llm"))

        # fallback: run sync in executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.base.score(query, doc))

    async def score(self, query: str, doc: str) -> float:
        text = await self._call_llm_async(query, doc)
        return _safe_clip(self.base._extract_score(text), 0.0, 1.0)

    async def score_batch(self, query: str, docs: List[str]) -> List[float]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(d):
            async with sem:
                return await self.score(query, d)

        return list(await asyncio.gather(*[_one(d) for d in docs]))


# ---------------------------
# Cached rerankers
# ---------------------------

@dataclass
class CachedLLMReranker(LLMReranker):
    cache: Dict[Tuple[int, int], float] = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = {}

    def score(self, query: str, doc: str) -> float:
        key = (hash(query), hash(doc))
        if key in self.cache:
            return self.cache[key]
        s = super().score(query, doc)
        self.cache[key] = s
        return s


@dataclass
class AsyncCachedLLMReranker(AsyncLLMReranker):
    cache: Dict[Tuple[int, int], float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.cache is None:
            self.cache = {}

    async def score(self, query: str, doc: str) -> float:
        key = (hash(query), hash(doc))
        if key in self.cache:
            return self.cache[key]
        s = await super().score(query, doc)
        self.cache[key] = s
        return s


# ---------------------------
# RerankerFactory
# ---------------------------

class RerankerFactory:
    """
    Choose between CrossEncoder and LLM reranker.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        cross_model: str = "BAAI/bge-reranker-base",
        llm_threshold: int = 30,
        use_cache: bool = True,
    ):
        self.llm = llm
        self.cross_model = cross_model
        self.llm_threshold = llm_threshold
        self.use_cache = use_cache
        self._cache = {}

    def create(self, top_k: int):
        if self.llm is not None and top_k <= self.llm_threshold:
            if self.use_cache:
                return CachedLLMReranker(llm=self.llm, cache=self._cache)
            return LLMReranker(llm=self.llm)

        return CrossEncoderReranker(model_name=self.cross_model)


# ---------------------------
# Unified rerank interface
# ---------------------------

def rerank_candidates(
    query: str,
    candidates: Sequence[TextLike],
    reranker: BaseReranker,
    *,
    top_n: int,
    content_key: str = "text",
    normalize: str = "minmax",
    sigmoid_temperature: float = 1.0,
    include_debug: bool = False,
) -> List[Tuple[TextLike, RerankResult]]:

    if top_n <= 0:
        return []

    docs = [_to_doc_text(c, content_key=content_key) for c in candidates]
    raw = reranker.score_batch(query, docs)

    if normalize == "none":
        norm = list(raw)
    elif normalize == "sigmoid":
        norm = sigmoid_calibrate(raw, temperature=sigmoid_temperature)
    else:
        norm = minmax_normalize(raw)

    results = []
    for c, rs, ns in zip(candidates, raw, norm):
        meta = {"raw": rs, "norm": ns} if include_debug else {}
        results.append((c, RerankResult(raw_score=rs, norm_score=ns, meta=meta)))

    results.sort(key=lambda x: x[1].norm_score, reverse=True)
    return results[:top_n]
