from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import json
import re

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


SYSTEM_SPLIT = "========================================================================\nUSER MESSAGE"

EXAMPLE_BLOCK_RE = re.compile(
    r"(?ms)^\[EXAMPLE\s*\|\s*(?P<qt>[A-Z_\-]+)\]\s*\n(?P<body>.*?)(?=^\[EXAMPLE\s*\||^\Z)"
)


def _normalize_qt(qt: str) -> str:
    return (qt or "").strip().upper().replace("-", "_")


def _split_system_user(prompt_text: str) -> Tuple[str, str]:
    """
    Split prompt file into system and user sections.
    If no split marker found, treat entire file as system, empty user.
    """
    if SYSTEM_SPLIT in prompt_text:
        sys_part, user_part = prompt_text.split(SYSTEM_SPLIT, 1)
        user_part = "USER MESSAGE" + user_part
        return sys_part.strip(), user_part.strip()
    return prompt_text.strip(), ""


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


def _select_one_example(examples_region: str, query_type: str) -> str:
    """
    Pick ONE example block matching query_type. Fallback to OTHER or first.
    Example block format:
      [EXAMPLE | PERFORMANCE]
      ...
    """
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

        # Prompt: system/user split + example selection
        self.prompt_path = Path("legalrag/prompts/legal_rag_prompt.txt")
        prompt_text = self.prompt_path.read_text(encoding="utf-8")

        self._sys_part, self._user_part = _split_system_user(prompt_text)
        self._user_prefix, self._user_examples, self._user_suffix = _extract_user_sections(self._user_part)

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
        # Context block
        law_context_parts: List[str] = []
        for i, h in enumerate(hits, start=1):
            c = getattr(h, "chunk", None)
            if c is None:
                continue
            law_context_parts.append(
                f"[候选条文 {i}]\n"
                f"条号：{getattr(c, 'article_no', '') or ''}\n"
                f"章节：{(getattr(c, 'chapter', '') or '')} {(getattr(c, 'section', '') or '')}\n"
                f"内容：{(getattr(c, 'text', '') or '').strip()}\n"
            )
        law_context = "\n\n".join(law_context_parts) if law_context_parts else "（当前未检索到相关条文）"

        # Pick ONE few-shot example based on query type
        one_example = _select_one_example(self._user_examples, query_type)

        # Compile user message
        user_compiled = "\n\n".join([s for s in [self._user_prefix, one_example, self._user_suffix] if s.strip()])
        user_compiled = user_compiled.format(question=question, query_type=query_type, law_context=law_context)

        return [
            {"role": "system", "content": self._sys_part.strip()},
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
        llm = llm_override or self.llm

        try:
            raw = llm.chat(messages=messages)
        except TypeError:
            # If client only supports prompt
            raw = llm.chat(prompt=self._messages_to_prompt(messages))

        return RagAnswer(question=question, answer=_trim_to_answer(raw), hits=hits)

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
