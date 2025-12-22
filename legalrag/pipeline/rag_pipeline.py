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
from legalrag.schemas import LawChunk, QueryType, RagAnswer, RetrievalHit, RoutingMode
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


def extract_key_term(question: str) -> str:
    q = question.strip().rstrip("？?")
    for p in ("什么是", "如何理解", "本法所称", "本条所称", "本编所称"):
        if q.startswith(p):
            return q[len(p):].strip()
    return q

 
EXAMPLE_BLOCK_RE = re.compile(
    r"(?ms)^\[EXAMPLE\s*\|\s*(?P<qt>[A-Z_\-]+)\]\s*\n(?P<body>.*?)(?=^\[EXAMPLE\s*\||^\Z)"
)


def _normalize_qt(qt: str) -> str:
    return (qt or "").strip().upper().replace("-", "_")


def _extract_user_sections(user_part: str) -> Tuple[str, str, str]:
    """
    Expect:
      A) FORMAT EXAMPLES
      B) NOW ANSWER THE REAL QUESTION
    Return (prefix_before_A, examples_region, suffix_from_B)
    """
    a_idx = user_part.find("A) FORMAT EXAMPLES")
    b_idx = user_part.find("B) NOW ANSWER THE REAL QUESTION")
    if a_idx == -1 or b_idx == -1:
        return "", user_part, ""
    prefix = user_part[:a_idx].rstrip()
    examples = user_part[a_idx:b_idx].strip()
    suffix = user_part[b_idx:].lstrip()
    return prefix, examples, suffix


def _select_one_example(examples_region: Any, query_type: str) -> str:
    """
    Pick ONE example block matching query_type. Fallback to OTHER or first.
    Example block format:
      [EXAMPLE | PERFORMANCE]
      ...
    """
    # If examples are provided as a dict, pick by normalized query type.
    if isinstance(examples_region, dict):
        qt = _normalize_qt(query_type)
        # try exact match, then OTHER, then any
        if qt in examples_region and str(examples_region.get(qt)).strip():
            return str(examples_region.get(qt)).strip()
        if "OTHER" in examples_region and str(examples_region.get("OTHER")).strip():
            return str(examples_region.get("OTHER")).strip()
        for v in examples_region.values():
            if str(v).strip():
                return str(v).strip()
        return ""

    qt = _normalize_qt(query_type)
    blocks: Dict[str, str] = {}
    for m in EXAMPLE_BLOCK_RE.finditer(examples_region):
        blocks[_normalize_qt(m.group("qt"))] = m.group(0).strip()

    if not blocks:
        return ""
    if qt in blocks:
        return blocks[qt]
    if "OTHER" in blocks:
        return blocks["OTHER"]
    return next(iter(blocks.values()))


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
        self.graph = LawGraphStore(cfg)

        # LLM
        self.llm = LLMClient.from_config(cfg)

        # Router
        self.router = QueryRouter(llm_client=self.llm, llm_based=cfg.routing.llm_based)

        # Embedder for graph-aware rerank
        model_name = cfg.retrieval.embedding_model
        logger.info(f"[RAG] Loading embedding model for graph-aware rerank: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embed_model.eval()

        # prov = (getattr(self.llm, "provider", "") or "").strip().lower() 
        self.zh_prompt_path = Path("legalrag/prompts/prompt_zh.json")
        self.en_prompt_path = Path("legalrag/prompts/prompt_en.json")

        zh_prompt_obj = json.loads(self.zh_prompt_path.read_text(encoding="utf-8"))
        self._zh_sys_part = (zh_prompt_obj.get("system") or "").strip()
        self._zh_user_prefix = (zh_prompt_obj.get("user_prefix") or "").strip()
        self._zh_user_suffix = (zh_prompt_obj.get("user_suffix") or "").strip()
        self._zh_user_examples = zh_prompt_obj.get("examples") or {}
        en_prompt_obj = json.loads(self.en_prompt_path.read_text(encoding="utf-8"))
        self._en_sys_part = (en_prompt_obj.get("system") or "").strip()
        self._en_user_prefix = (en_prompt_obj.get("user_prefix") or "").strip()
        self._en_user_suffix = (en_prompt_obj.get("user_suffix") or "").strip()
        self._en_user_examples = en_prompt_obj.get("examples") or {}

    # -----------------------
    # Embedding
    # -----------------------
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype="float32")

        with torch.no_grad():
            inputs = self.embed_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.embed_model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-8)
            emb = (summed / counts).cpu().numpy().astype("float32")

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norms

    # -----------------------
    # Graph-aware retrieval
    # -----------------------
    def _graph_augmented_retrieval(self, question: str, decision: Any, top_k: int) -> List[RetrievalHit]:

        rag_hits = self.retriever.search(question, top_k=top_k)
        for h in rag_hits:
            h.source = "retriever"

        seed_ids = [getattr(h.chunk, "id", None) or getattr(h.chunk, "article_id", None) for h in rag_hits]
        seed_ids = [sid for sid in seed_ids if sid]

        graph_nodes: List[Any] = []
        if hasattr(self.graph, "walk"):
            try:
                graph_nodes = self.graph.walk(
                    start_ids=seed_ids,
                    depth=2,
                    rel_types=["refers_to", "related", "same_topic", "parent_of"],
                    limit=80,
                )
            except TypeError:
                graph_nodes = self.graph.walk(start_ids=seed_ids, depth=2)
        elif hasattr(self.graph, "get_neighbors"):
            for sid in seed_ids:
                try:
                    graph_nodes.extend(self.graph.get_neighbors(sid, depth=2))
                except TypeError:
                    graph_nodes.extend(self.graph.get_neighbors(sid))

        if getattr(decision, "query_type", None) == QueryType.DEFINITION and hasattr(self.graph, "get_definition_sources"):
            term = extract_key_term(question)
            try:
                graph_nodes.extend(self.graph.get_definition_sources(term))
            except Exception:
                logger.warning("[RAG][graph] get_definition_sources failed", exc_info=True)

        seen_ids = set(seed_ids)
        unique_nodes: List[Any] = []
        for n in graph_nodes:
            aid = getattr(n, "article_id", None) or getattr(n, "id", None)
            if not aid or aid in seen_ids:
                continue
            seen_ids.add(aid)
            unique_nodes.append(n)

        graph_hits: List[RetrievalHit] = []
        for n in unique_nodes:
            chunk = LawChunk(
                id=getattr(n, "article_id", None) or getattr(n, "id", None),
                law_name=getattr(n, "law_name", "民法典·合同编"),
                chapter=getattr(n, "chapter", None),
                section=getattr(n, "section", None),
                article_no=getattr(n, "article_no", ""),
                text=getattr(n, "text", ""),
                source="graph",
                start_char=None,
                end_char=None,
            )
            graph_hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=0.5,
                    rank=-1,
                    source="graph",
                    graph_depth=getattr(n, "graph_depth", None),
                    relations=getattr(n, "relations", None),
                )
            )

        logger.info("len(graph_hits)=", len(graph_hits))
        combined: List[RetrievalHit] = rag_hits + graph_hits
        if not combined:
            return []

        vecs = self._encode_texts([h.chunk.text or "" for h in combined])
        qvec = self._encode_texts([question])[0]

        for h, v in zip(combined, vecs):
            sem = cosine_similarity(qvec, v)
            h.semantic_score = sem
            h.score = 0.6 * float(h.score) + 0.4 * sem

        combined.sort(key=lambda h: h.score, reverse=True)
        top = combined[:top_k]
        for i, h in enumerate(top, start=1):
            h.rank = i
        return top

    # -----------------------
    # Prompt -> messages
    # -----------------------
    def _build_messages(self, question: str, hits: List[RetrievalHit], query_type: str) -> List[Dict[str, str]]:

        def is_chinese(text: str) -> bool:
            """Simple check: returns True if text contains at least one Chinese character."""
            return bool(re.search(r'[\u4e00-\u9fff]', text))
        def _build_law_context_part(c, i, use_chinese):
            if use_chinese:
                # 中文版本
                return (
                    f"[候选条文 {i}]\n"
                    f"章节：{(getattr(c, 'chapter', '') or '')} {(getattr(c, 'section', '') or '')}\n"
                    f"条号：{getattr(c, 'article_no', '') or ''}\n"
                    f"内容：{(getattr(c, 'text', '') or '').strip()}\n"
                )
            else:
                # 英文版本
                return (
                    f"[Candidate Provision {i}]\n"
                    f"Chapter/Section: {(getattr(c, 'chapter', '') or '')} {(getattr(c, 'section', '') or '')}\n"
                    f"Article: {getattr(c, 'article_no', '') or ''}\n"
                    f"Text: {(getattr(c, 'text', '') or '').strip()}\n"
                )

        is_chinese_question = is_chinese(question)

        law_context_parts: List[str] = []
        for i, h in enumerate(hits, start=1):
            c = getattr(h, "chunk", None)
            if c is None:
                continue
            law_context_parts.extend(_build_law_context_part(c, i, is_chinese_question) )


        # Choose user prefix
        _user_prefix = self._zh_user_prefix if is_chinese_question else self._en_user_prefix
        _user_suffix = self._zh_user_suffix if is_chinese_question else self._en_user_suffix
        _sys_part = self._zh_sys_part if is_chinese_question else self._en_sys_part
        _user_examples = self._zh_user_examples if is_chinese_question else self._en_user_examples


        law_context = "\n\n".join(law_context_parts) if law_context_parts else ""

        # Pick ONE few-shot example based on query type
        one_example = _select_one_example(_user_examples, query_type)

        # Compile user message
        user_compiled = "\n\n".join([s for s in [_user_prefix, _user_suffix, one_example] if s.strip()])
        user_compiled = user_compiled.format(question=question, query_type=query_type, law_context=law_context)

        logger.info("[_build_messages] query_type=%s", query_type)
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
    def retrieve(self, question: str, top_k: Optional[int] = None) -> Tuple[Any, List[RetrievalHit], int]:
        decision = self.router.route(question)

        base_k = top_k or self.cfg.retrieval.top_k
        eff_top_k = int(base_k * getattr(decision, "top_k_factor", 1.0))
        eff_top_k = max(3, min(eff_top_k, 30))

        if getattr(decision, "mode", None) == RoutingMode.GRAPH_AUGMENTED:
            hits = self._graph_augmented_retrieval(question, decision, eff_top_k)
        else:
            hits = self.retriever.search(question, top_k=eff_top_k)
            for h in hits:
                h.source = "retriever"

        return decision, hits, eff_top_k

    def answer_from_hits(
        self,
        question: str,
        hits: List[RetrievalHit],
        decision: Optional[Any] = None,
        llm_override: Optional[LLMClient] = None,
    ) -> RagAnswer:
        qt = getattr(decision, "query_type", None) if decision is not None else None
        qt_str = str(qt) if qt is not None else "OTHER"

        messages = self._build_messages(question, hits, qt_str)
        # logger.info("[answer_from_hits] messages=%s", messages)
        llm = llm_override or self.llm

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

        query_type = None
        if decision is not None:
            query_type = getattr(decision, "query_type", None)  
        else:
            logger.info("[answer_stream_from_hits] decision is None")
            query_type = "OTHER"

        try:
            messages = self._build_messages(question, hits, query_type)
        except TypeError:
            messages = self._build_messages(question, hits)
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

        # Branch 1: async iterator/generator (OpenAI in your current implementation)
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
        decision, hits, eff_top_k = self.retrieve(question, top_k=top_k)
        logger.info(
            "[RAG] query: %s; query_type=%s mode=%s top_k=%d",
            question,
            getattr(decision, "query_type", None),
            getattr(decision, "mode", None),
            eff_top_k,
        )
        return self.answer_from_hits(question, hits, decision=decision, llm_override=llm_override)