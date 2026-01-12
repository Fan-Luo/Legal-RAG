from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import json
import re

import asyncio
import inspect

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.schemas import IssueType, LawChunk, RagAnswer, RetrievalHit, TaskType
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.retrieval.graph_store import LawGraphStore
from legalrag.routing.router import QueryRouter
from legalrag.utils.logger import get_logger
from typing import AsyncIterator
import time
import threading
start = time.time()


logger = get_logger(__name__)


# -----------------------
# Utilities
# -----------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def _normalize_tag(tag: str) -> str:
    return (tag or "").strip().lower()


def _select_one_example(
    example_pool: Any,
    *,
    lang: str,
    task_type: str,
    issue_type: str,
) -> str:
    """
    Pick ONE example block based on language + tags.
    """
    if not isinstance(example_pool, list):
        return ""

    lang_key = (lang or "").strip().lower()
    task_tag = f"task:{(task_type or '').strip().lower()}"
    issue_tag = f"issue:{(issue_type or '').strip().lower()}"

    candidates = [e for e in example_pool if str(e.get("lang", "")).strip().lower() == lang_key]
    if not candidates:
        candidates = list(example_pool)

    best = ""
    best_score = -1
    for e in candidates:
        tags = {_normalize_tag(t) for t in (e.get("tags") or [])}
        score = 0
        if task_tag in tags:
            score += 3
        if issue_tag in tags:
            score += 2
        if score > best_score and str(e.get("content", "")).strip():
            best_score = score
            best = str(e.get("content", "")).strip()

    return best


def _trim_to_answer(text: str) -> str:
    """
    Try to cut off any accidental prefix; keep from the first '结论：' if present.
    """
    if not isinstance(text, str):
        text = str(text)
    idx = text.find("结论：")
    return text[idx:].strip() if idx != -1 else text.strip()



class RagPipeline:
    """
    RagPipeline with two-stage support:
      - retrieve(question, top_k=None) -> (decision, hits, eff_top_k)
      - answer_from_hits(question, hits, decision=None, llm_override=None) -> RagAnswer

    Keep the single method answer(question, top_k=None, llm_override=None) -> RagAnswer
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # Retrieval components
        self.retriever = HybridRetriever(cfg)

        # LLM
        self.llm = LLMClient.from_config(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prompt
        self.zh_prompt_path = Path("legalrag/prompts/prompt_zh.json")
        self.en_prompt_path = Path("legalrag/prompts/prompt_en.json")
        zh_prompt_obj = json.loads(self.zh_prompt_path.read_text(encoding="utf-8"))
        self._zh_registry = zh_prompt_obj.get("registry") or {}
        self._zh_example_pool = zh_prompt_obj.get("example_pool") or []
        self._zh_defaults = zh_prompt_obj.get("defaults") or {}
        en_prompt_obj = json.loads(self.en_prompt_path.read_text(encoding="utf-8"))
        self._en_registry = en_prompt_obj.get("registry") or {}
        self._en_example_pool = en_prompt_obj.get("example_pool") or []
        self._en_defaults = en_prompt_obj.get("defaults") or {}

    # -----------------------
    # Prompt -> messages
    # -----------------------
    def _build_messages(
        self,
        question: str,
        hits: List[RetrievalHit],
        task_type: TaskType | str,
        issue_type: IssueType | str,
    ) -> List[Dict[str, str]]:

        def is_chinese(text: str) -> bool:
            """Simple check: returns True if text contains at least one Chinese character."""
            return bool(re.search(r'[\u4e00-\u9fff]', text))
        def _enum_val(v: Any) -> str:
            return str(getattr(v, "value", v or "")).strip()
        def _compose_suffix(tpl: Dict[str, Any]) -> str:
            parts = []
            for key in ("output_structure", "citation_rules", "format_constraints", "forbidden", "user_suffix"):
                val = str(tpl.get(key) or "").strip()
                if val:
                    parts.append(val)
            return "\n\n".join(parts).strip()
        def _build_law_context_part(c, i, use_chinese):
            if use_chinese:
                # 中文
                return (
                    f"[候选条文 {i}]\n"
                    f"章节：{(getattr(c, 'chapter', '') or '')} {(getattr(c, 'section', '') or '')}\n"
                    f"条号：{getattr(c, 'article_id', '') or ''}\n"
                    f"内容：{(getattr(c, 'text', '') or '').strip()}\n"
                )
            else:
                # 英文
                return (
                    f"[Candidate Provision {i}]\n"
                    f"Chapter/Section: {(getattr(c, 'chapter', '') or '')} {(getattr(c, 'section', '') or '')}\n"
                    f"Article: {getattr(c, 'article_id', '') or ''}\n"
                    f"Text: {(getattr(c, 'text', '') or '').strip()}\n"
                )

        is_chinese_question = is_chinese(question)

        law_context_parts: List[str] = []
        for i, h in enumerate(hits, start=1):
            c = getattr(h, "chunk", None)
            if c is None:
                continue
            law_context_parts.append(_build_law_context_part(c, i, is_chinese_question) )


        # Choose registry + example pool
        registry = self._zh_registry if is_chinese_question else self._en_registry
        example_pool = self._zh_example_pool if is_chinese_question else self._en_example_pool
        defaults = self._zh_defaults if is_chinese_question else self._en_defaults

        task_key = _enum_val(task_type).lower() or str(defaults.get("task_type", TaskType.JUDGE_STYLE.value)).lower()
        issue_key = _enum_val(issue_type).lower() or IssueType.OTHER.value
        default_task = str(defaults.get("task_type", TaskType.JUDGE_STYLE.value)).lower()

        tpl = registry.get(task_key) or registry.get(default_task) or {}
        _sys_part = str(tpl.get("system") or "").strip()
        _user_prefix = str(tpl.get("user_prefix") or "").strip()
        _user_suffix = _compose_suffix(tpl)


        law_context = "\n\n".join(law_context_parts) if law_context_parts else ""

        # Pick ONE few-shot example based on task_type + issue_type + lang
        one_example = _select_one_example(
            example_pool,
            lang="zh" if is_chinese_question else "en",
            task_type=task_key,
            issue_type=issue_key,
        )

        # Compile user message
        user_compiled = "\n\n".join([s for s in [_user_prefix, _user_suffix, one_example] if s.strip()])
        user_compiled = user_compiled.format(
            question=question,
            task_type=task_key,
            issue_type=issue_key,
            law_context=law_context,
        )

        # logger.info("[_build_messages] task_type=%s issue_type=%s", task_key, issue_key)
        return [
            {"role": "system", "content": _sys_part.strip()},
            {"role": "user", "content": user_compiled.strip()},
        ]

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # for LLM clients that only accept prompt.
        chunks: List[str] = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            chunks.append(f"{role}:\n{m.get('content','')}")
        return "\n\n".join(chunks)

    # -----------------------
    # Two-stage public API
    # -----------------------
    def retrieve(self, question: str, llm_override: Optional[LLMClient] = None, top_k: Optional[int] = None, decision: Optional[Any] = None) -> Tuple[Any, List[RetrievalHit], int]:
        llm = llm_override or self.llm
        if decision is None:
            self.router = QueryRouter(llm_client=llm, llm_based=self.cfg.routing.llm_based)
            decision = self.router.route(question)

        base_k = top_k or self.cfg.retrieval.top_k
        eff_top_k = int(base_k * getattr(decision, "top_k_factor", 1.0))
        eff_top_k = max(3, min(eff_top_k, 30))

        hits = self.retriever.search(question, llm=llm, top_k=eff_top_k, decision=decision)
       
        return decision, hits, eff_top_k

    def answer_from_hits(
        self,
        question: str,
        hits: List[RetrievalHit],
        decision: Optional[Any] = None,
        llm: Optional[LLMClient] = None,
    ) -> RagAnswer:
        task = getattr(decision, "task_type", None) if decision is not None else TaskType.JUDGE_STYLE
        issue = getattr(decision, "issue_type", None) if decision is not None else IssueType.OTHER
        messages = self._build_messages(question, hits, task, issue)
        # logger.info("[answer_from_hits] messages=%s", messages)

        try:
            raw = llm.chat(messages=messages)
        except TypeError:
            # If client only supports prompt
            raw = llm.chat(prompt=self._messages_to_prompt(messages))

        return RagAnswer(question=question, answer=_trim_to_answer(raw), hits=hits)

    async def answer_stream_from_hits(
        self,
        question: str,
        hits: list,
        decision: Optional[Any] = None,
        llm_override: Optional[dict] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        t_build0 = time.time()

        task = getattr(decision, "task_type", None) if decision is not None else TaskType.JUDGE_STYLE
        issue = getattr(decision, "issue_type", None) if decision is not None else IssueType.OTHER
        messages = self._build_messages(question, hits, task, issue)
        # logger.info("[answer_stream_from_hits] messages=%s", messages)
        logger.info("[TIMING] build_messages=%.3fs", time.time() - t_build0)

        llm = llm_override or self.llm
        if llm is None:
            raise RuntimeError("LLM not initialized")

        stream_fn = getattr(llm, "chat_stream", None)
        if not callable(stream_fn):
            raise RuntimeError("LLM chat_stream() not available; cannot stream")

        t_call0 = time.time()
        stream_obj = stream_fn(messages)
        logger.info("[TIMING] chat_stream_call=%.3fs", time.time() - t_call0)

        # Branch 1: async iterator/generator 
        if hasattr(stream_obj, "__aiter__"):
            first = True
            async for piece in stream_obj:
                if first:
                    logger.info("[TIMING] first_piece_after_call=%.3fs", time.time() - t_call0)
                    first = False
                if piece:
                    yield piece
            return

        # Branch 2: sync iterator -> thread bridge -> async yield
        q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        first_sent = False

        def _worker():
            nonlocal first_sent
            try:
                for piece in stream_obj:
                    if piece:
                        if not first_sent:
                            logger.info("[TIMING] first_piece_after_call=%.3fs", time.time() - t_call0)
                            first_sent = True
                        asyncio.run_coroutine_threadsafe(q.put(piece), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    def answer(self, question: str, top_k: Optional[int] = None, llm_override: Optional[LLMClient] = None) -> RagAnswer:
        llm = llm_override or self.llm
        decision, hits, eff_top_k = self.retrieve(question, llm, top_k=top_k)
        logger.info(
            "[RAG] query: %s; task_type=%s issue_type=%s mode=%s top_k=%d",
            question,
            getattr(decision, "task_type", None),
            getattr(decision, "issue_type", None),
            getattr(decision, "mode", None),
            eff_top_k,
        )
        return self.answer_from_hits(question, hits, decision=decision, llm=llm)
