from __future__ import annotations

""" 
Works with local rerankers (CrossEncoder) and LLM-based rerankers.

Key abstractions:
- BaseReranker: synchronous reranker interface (score_batch preferred).
- AsyncRerankerMixin: optional async scoring for LLM calls.
- RerankResult: normalized + raw scores for blending / debugging.

Default:
- CrossEncoderReranker(model_name="BAAI/bge-reranker-base" or "BAAI/bge-reranker-large")

Optional:
- LLMReranker for small top_n (e.g., 10-30), or as label generator for LTR.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

import math
import re
import time


# ---------------------------
# Types
# ---------------------------

TextLike = Union[str, Dict[str, Any]]


class BaseReranker(Protocol):
    """Synchronous reranker interface."""

    def score(self, query: str, doc: str) -> float: ...

    def score_batch(self, query: str, docs: List[str]) -> List[float]:
        """Preferred for efficiency. Implementations should override when possible."""
        return [self.score(query, d) for d in docs]


class AsyncReranker(Protocol):
    """Optional async interface (useful for LLM reranking)."""

    async def score(self, query: str, doc: str) -> float: ...

    async def score_batch(self, query: str, docs: List[str]) -> List[float]: ...


@dataclass(frozen=True)
class RerankResult:
    """Reranking output for a single candidate."""
    raw_score: float
    norm_score: float
    meta: Dict[str, Any]


# ---------------------------
# Normalization / calibration
# ---------------------------

def minmax_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def zscore_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    m = sum(scores) / len(scores)
    var = sum((s - m) ** 2 for s in scores) / max(1, (len(scores) - 1))
    sd = math.sqrt(var) if var > 1e-12 else 0.0
    if sd < 1e-12:
        return [0.0 for _ in scores]
    return [(s - m) / sd for s in scores]


def sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_calibrate(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    t = max(1e-6, float(temperature))
    return [sigmoid(s / t) for s in scores]


# ---------------------------
# Helpers
# ---------------------------

def _to_doc_text(doc: TextLike, content_key: str = "text") -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        # common keys in IR pipelines
        for k in (content_key, "text", "content", "passage", "chunk", "body"):
            if k in doc and isinstance(doc[k], str):
                return doc[k]
        return str(doc)
    return str(doc)


def _safe_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------
# Cross-encoder reranker
# ---------------------------

@dataclass
class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers CrossEncoder.
      - Produces "raw scores" in model scale,
      - Caller can normalize via `rerank_candidates()` using min-max/sigmoid, etc.
    """

    model_name: str = "BAAI/bge-reranker-base"
    device: Optional[str] = None
    max_length: int = 512
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "CrossEncoderReranker requires `sentence-transformers`. "
                "Install with: pip install sentence-transformers"
            ) from e

        kwargs: Dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, **kwargs)

    def score(self, query: str, doc: str) -> float:
        return float(
            self._model.predict([(query, doc)], batch_size=1, max_length=self.max_length)[0]
        )

    def score_batch(self, query: str, docs: List[str]) -> List[float]:
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(
            pairs,
            batch_size=int(self.batch_size),
            max_length=self.max_length,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


# ---------------------------
# LLM reranker (sync + async)
# ---------------------------

@dataclass
class LLMReranker:
    """
    LLM-as-reranker (synchronous API).

    This expects an `llm` adapter that can be called in ONE of these ways:
      1) llm.chat(messages=[...], **kwargs) -> str
      2) llm.complete(prompt=..., system=..., **kwargs) -> str
      3) callable(system_prompt: str, user_prompt: str) -> str

    Guardrails:
      - temperature defaults to 0
      - score range is clipped to [0, 1]
      - output parsing is robust (JSON-first, then regex)

    Performance:
      - Use only on small top_n (e.g., 10-30).
      - For larger top_n, prefer CrossEncoder.
    """

    llm: Any
    model_name: Optional[str] = None
    temperature: float = 0.0

    # hard limits for prompt safety
    max_query_chars: int = 800
    max_doc_chars: int = 2000

    system_prompt: str = (
        "You are a ranking assistant. Given a query and a candidate evidence snippet, "
        "output ONLY a compact JSON object: {\"score\": <float 0..1>, \"rationale\": \"...\"}. "
        "Score=1 means directly answers the query; 0 means irrelevant. "
        "Do not include any other keys or text."
    )

    def _truncate(self, s: str, n: int) -> str:
        s = s or ""
        return s if len(s) <= n else s[:n] + "…"

    def _build_user_prompt(self, query: str, doc: str) -> str:
        q = self._truncate(query, self.max_query_chars)
        d = self._truncate(doc, self.max_doc_chars)
        return (
            f"Query:\n{q}\n\n"
            f"Candidate:\n{d}\n\n"
            "Return JSON only: {\"score\": <float 0..1>, \"rationale\": \"...\"}"
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        # 1) chat(messages=[...]) style
        if hasattr(self.llm, "chat") and callable(getattr(self.llm, "chat")):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return str(self.llm.chat(messages=messages, model=self.model_name, temperature=self.temperature))

        # 2) complete(prompt=..., system=...) style
        if hasattr(self.llm, "complete") and callable(getattr(self.llm, "complete")):
            return str(self.llm.complete(system=system_prompt, prompt=user_prompt, model=self.model_name, temperature=self.temperature))

        # 3) callable adapter
        if callable(self.llm):
            return str(self.llm(system_prompt, user_prompt))

        raise RuntimeError("LLMReranker: unsupported llm adapter; provide llm.chat / llm.complete / callable.")

    def score(self, query: str, doc: str) -> float:
        user_prompt = self._build_user_prompt(query, doc)
        text = self._call_llm(self.system_prompt, user_prompt)
        return _safe_clip(self._extract_score(text), 0.0, 1.0)

    def score_batch(self, query: str, docs: List[str]) -> List[float]:
        # Default sequential (safe). If you need concurrency, use AsyncLLMReranker below.
        return [self.score(query, d) for d in docs]

    @staticmethod
    def _extract_score(text: str) -> float:
        import json

        t = (text or "").strip()

        # Prefer JSON
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception:
            pass

        # Allow fenced code blocks with JSON
        m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "score" in obj:
                    return float(obj["score"])
            except Exception:
                pass

        # Regex fallbacks
        m = re.search(r"score\s*[:=]\s*([0-1](?:\.\d+)?)", t, re.IGNORECASE)
        if m:
            return float(m.group(1))

        # Any float 0..1
        m = re.search(r"([0-1](?:\.\d+)?)", t)
        if m:
            return float(m.group(1))

        return 0.0


@dataclass
class AsyncLLMReranker:
    """
    LLM-as-reranker with asyncio concurrency.

    The `llm` adapter must expose ONE of:
      - async llm.achat(messages=[...], **kwargs) -> str
      - async llm.acomplete(system=..., prompt=..., **kwargs) -> str
      - or a synchronous adapter is accepted but will run in threadpool (slower).

    This is intended for:
      - small candidate sets (10-50), where latency matters
      - strict concurrency limits to avoid rate-limit bursts
    """

    llm: Any
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_concurrency: int = 8

    # reuse prompts & parsing from LLMReranker
    base: LLMReranker | None = None

    def __post_init__(self) -> None:
        if self.base is None:
            self.base = LLMReranker(
                llm=self.llm,
                model_name=self.model_name,
                temperature=self.temperature,
            )

    async def _call_llm_async(self, system_prompt: str, user_prompt: str) -> str:
        # async chat
        if hasattr(self.llm, "achat") and callable(getattr(self.llm, "achat")):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return str(await self.llm.achat(messages=messages, model=self.model_name, temperature=self.temperature))

        # async complete
        if hasattr(self.llm, "acomplete") and callable(getattr(self.llm, "acomplete")):
            return str(await self.llm.acomplete(system=system_prompt, prompt=user_prompt, model=self.model_name, temperature=self.temperature))

        # fallback: run sync in executor
        import asyncio
        loop = asyncio.get_running_loop()
        return str(await loop.run_in_executor(None, lambda: self.base._call_llm(system_prompt, user_prompt)))  # type: ignore

    async def score(self, query: str, doc: str) -> float:
        assert self.base is not None
        user_prompt = self.base._build_user_prompt(query, doc)
        text = await self._call_llm_async(self.base.system_prompt, user_prompt)
        return _safe_clip(self.base._extract_score(text), 0.0, 1.0)

    async def score_batch(self, query: str, docs: List[str]) -> List[float]:
        import asyncio

        sem = asyncio.Semaphore(max(1, int(self.max_concurrency)))

        async def _one(d: str) -> float:
            async with sem:
                return await self.score(query, d)

        tasks = [_one(d) for d in docs]
        return list(await asyncio.gather(*tasks))


# ---------------------------
# Rerank orchestration utilities
# ---------------------------

def rerank_candidates(
    query: str,
    candidates: Sequence[TextLike],
    reranker: BaseReranker,
    *,
    top_n: int,
    content_key: str = "text",
    normalize: str = "minmax",  # "minmax" | "sigmoid" | "none"
    sigmoid_temperature: float = 1.0,
    include_debug: bool = False,
) -> List[Tuple[TextLike, RerankResult]]:
    """
    Rerank `candidates` and return top_n in descending relevance.

    - candidates can be strings or dict-like chunks.
    - normalization is applied to reranker outputs for downstream blending.

    Returns:
      [(candidate, RerankResult), ...] sorted by norm_score desc
    """
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

    results: List[Tuple[TextLike, RerankResult]] = []
    for c, rs, ns in zip(candidates, raw, norm):
        meta = {}
        if include_debug:
            meta = {"raw": rs, "norm": ns}
        results.append((c, RerankResult(raw_score=float(rs), norm_score=float(ns), meta=meta)))

    results.sort(key=lambda x: x[1].norm_score, reverse=True)
    return results[: int(top_n)]


def blend_scores(
    base_scores: Sequence[float],
    rerank_scores: Sequence[float],
    beta: float = 0.35,
) -> List[float]:
    """
    Linear blend of base_scores and rerank_scores in [0..1] scale:
      blended = (1-beta)*base + beta*rerank
    """
    b = _safe_clip(float(beta), 0.0, 1.0)
    n = min(len(base_scores), len(rerank_scores))
    return [(1.0 - b) * float(base_scores[i]) + b * float(rerank_scores[i]) for i in range(n)]
