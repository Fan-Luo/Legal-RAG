from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel

from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.schemas import (
    RagAnswer,
    RetrievalHit,
    QueryType,
    RoutingMode,
    LawChunk,
)
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.retrieval.graph_store import LawGraphStore
from legalrag.routing.router import QueryRouter
from legalrag.utils.logger import get_logger


logger = get_logger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Simple cosine similarity for 1D vectors."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def extract_key_term(question: str) -> str:
    """Heuristic keyword extraction for definition-type queries."""
    q = question.strip().rstrip("？?")

    patterns = ["什么是", "如何理解", "本法所称", "本条所称", "本编所称"]
    for p in patterns:
        if q.startswith(p):
            return q[len(p):].strip()

    return q

SYSTEM_SPLIT = "========================================================================\nUSER MESSAGE"
EXAMPLE_BLOCK_RE = re.compile(
    r"(?ms)^\[EXAMPLE\s*\|\s*(?P<qt>[A-Z_]+)\]\s*\n(?P<body>.*?)(?=^\[EXAMPLE\s*\||^\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-|\Z)"
)

def _normalize_qt(qt: str) -> str:
    return (qt or "").strip().upper().replace("-", "_")

def _split_system_user(prompt_text: str) -> Tuple[str, str]:
    """
    Split legal_rag_prompt.txt into system rules and user template.
    """
    if SYSTEM_SPLIT in prompt_text:
        sys_part, user_part = prompt_text.split(SYSTEM_SPLIT, 1)
        user_part = "USER MESSAGE" + user_part
        return sys_part.strip(), user_part.strip()
    return prompt_text.strip(), ""

def _extract_user_sections(user_part: str) -> Tuple[str, str, str]:
    """
    From USER MESSAGE, extract:
      - prefix: before examples section
      - examples region: includes all [EXAMPLE | ...] blocks
      - suffix: "B) NOW ANSWER..." section to end (contains placeholders + output anchor)
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
    Select exactly one [EXAMPLE | QUERYTYPE] block.
    Fallback:
      1) exact match
      2) OTHER
      3) first found
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
    Safety valve: keep from first "结论：" if present.
    """
    key = "结论："
    idx = text.find(key)
    return text[idx:].strip() if idx != -1 else text.strip()



class RagPipeline:
    """
    Graph-aware RAG Pipeline.
    - Build prompts as messages (system/user) to reduce instruction-echo.
    - Support llm_override per-request (e.g., user-provided OpenAI key).
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # Retrieval components
        self.retriever = HybridRetriever(cfg)
        self.graph = LawGraphStore(cfg)

        # LLM client (can be disabled / degraded, depending on cfg)
        self.llm = LLMClient.from_config(cfg)

        # Router (can be llm-based or heuristic depending on cfg.routing.llm_based)
        self.router = QueryRouter(
            llm_client=self.llm,
            llm_based=cfg.routing.llm_based,
        )

        # Embedder for graph-aware rerank (same embedding model as vector store)
        model_name = cfg.retrieval.embedding_model  # e.g. "BAAI/bge-base-zh-v1.5"
        logger.info(f"[RAG] Loading embedding model for graph-aware rerank: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embed_model.eval()

        self.prompt_path = Path("legalrag/prompts/legal_rag_prompt.txt")
        self._prompt_text = self.prompt_path.read_text(encoding="utf-8")
        self._sys_part, self._user_part = _split_system_user(self._prompt_text)
        self._user_prefix, self._user_examples, self._user_suffix = _extract_user_sections(self._user_part)

    # -----------------------
    # Embedding helper
    # -----------------------
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using BGE: mean pooling + L2 normalize."""
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
            last_hidden = outputs.last_hidden_state  # [b, seq, hidden]

            mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)  # [b, seq, 1]
            summed = (last_hidden * mask).sum(dim=1)                             # [b, hidden]
            counts = mask.sum(dim=1).clamp(min=1e-8)                             # [b, 1]
            emb = (summed / counts).cpu().numpy().astype("float32")

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norms

    # -----------------------
    # Graph-aware retrieval
    # -----------------------
    def _graph_augmented_retrieval(self, question: str, decision, top_k: int) -> List[RetrievalHit]:
        """Graph-aware retrieval: hybrid -> graph expansion -> semantic rerank."""
        rag_hits = self.retriever.search(question, top_k=top_k)
        for h in rag_hits:
            h.source = "retriever"

        seed_ids = [
            getattr(h.chunk, "id", None) or getattr(h.chunk, "article_id", None)
            for h in rag_hits
        ]
        seed_ids = [sid for sid in seed_ids if sid]

        graph_nodes = []
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

        # definition-special expansion
        if getattr(decision, "query_type", None) == QueryType.DEFINITION and hasattr(self.graph, "get_definition_sources"):
            term = extract_key_term(question)
            try:
                graph_nodes.extend(self.graph.get_definition_sources(term))
            except Exception:
                logger.warning("[RAG][graph] get_definition_sources failed", exc_info=True)

        # de-dup with baseline seeds
        seen_ids = set(seed_ids)
        unique_nodes = []
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

        texts = [h.chunk.text or "" for h in combined]
        vecs = self._encode_texts(texts)
        qvec = self._encode_texts([question])[0]

        for h, v in zip(combined, vecs):
            sem = cosine_similarity(qvec, v)
            h.semantic_score = sem
            h.score = 0.6 * float(h.score) + 0.4 * sem

        if getattr(decision, "query_type", None) == QueryType.DEFINITION:
            combined = self._boost_definition_hits(combined)

        combined.sort(key=lambda h: h.score, reverse=True)
        top = combined[:top_k]
        for i, h in enumerate(top, start=1):
            h.rank = i
            h.explain = {
                "source": getattr(h, "source", None),
                "semantic_score": getattr(h, "semantic_score", None),
                "graph_depth": getattr(h, "graph_depth", None),
                "relations": getattr(h, "relations", None),
            }
        return top

    def _boost_definition_hits(self, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """Boost definition-like hits by heuristics."""
        def is_definition_like(chunk: LawChunk) -> bool:
            text = (chunk.text or "").replace("\n", "")
            return any(p in text for p in ["是指", "本法所称", "本条所称", "本编所称"])

        boosted: List[RetrievalHit] = []
        for h in hits:
            bonus = 0.0
            if is_definition_like(h.chunk):
                bonus += 0.2
            if getattr(h.chunk, "chapter", None) and "总则" in (h.chunk.chapter or ""):
                bonus += 0.1
            rel = getattr(h, "relation", None)
            if rel in ("same_section", "sibling"):
                bonus += 0.05

            h.score = float(h.score) + bonus
            boosted.append(h)
        return boosted

    # ------------------------------
    # Messages builder (best-practice)
    # ------------------------------
    def _build_messages(
        self,
        question: str,
        hits: List[RetrievalHit],
        query_type: str,
    ) -> List[Dict[str, str]]:
        """
        Build system/user messages:
        - system: rules only
        - user: prefix + ONE example for query_type + suffix filled with runtime inputs
        """
        # evidence text (no score/source to reduce noise / instruction leakage)
        law_context_parts: List[str] = []
        for i, h in enumerate(hits, start=1):
            c = getattr(h, "chunk", None)
            if c is None:
                continue
            article_no = getattr(c, "article_no", "") or ""
            chapter = getattr(c, "chapter", "") or ""
            section = getattr(c, "section", "") or ""
            text = (getattr(c, "text", "") or "").strip()

            law_context_parts.append(
                f"[候选条文 {i}]\n"
                f"条号：{article_no}\n"
                f"章节：{chapter} {section}\n"
                f"内容：{text}\n"
            )

        law_context = "\n\n".join(law_context_parts) if law_context_parts else "（当前未检索到相关条文）"

        # choose exactly one example
        one_example = _select_one_example(self._user_examples, query_type)

        user_compiled = "\n\n".join([s for s in [self._user_prefix, one_example, self._user_suffix] if s.strip()])

        # fill placeholders
        user_compiled = user_compiled.format(
            question=question,
            query_type=query_type,
            law_context=law_context,
        )

        # Ensure the output anchor exists (safety)
        if "【OUTPUT — START HERE】" not in user_compiled and "结论：" not in user_compiled:
            user_compiled = user_compiled.rstrip() + "\n\n结论："

        return [
            {"role": "system", "content": self._sys_part.strip()},
            {"role": "user", "content": user_compiled.strip()},
        ]

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Compatibility fallback if a provider only accepts a single prompt string.
        """
        chunks = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            content = m.get("content") or ""
            chunks.append(f"{role}:\n{content}")
        return "\n\n".join(chunks).strip()

    # -----------------------
    # Public API
    # -----------------------
    def answer(self, question: str, top_k: Optional[int] = None, llm_override: Optional[LLMClient] = None) -> RagAnswer:
        """RAG entrypoint. Supports llm_override (per-request LLM client)."""
        decision = self.router.route(question)
        base_k = top_k or self.cfg.retrieval.top_k
        eff_top_k = int(base_k * getattr(decision, "top_k_factor", 1.0))
        eff_top_k = max(3, min(eff_top_k, 30))
        query_type = getattr(decision, 'query_type', None)
        query_type_str = str(query_type) if query_type is not None else "OTHER"

        logger.info(
            f"[RAG] query: {question}; , query_type: {query_type}"
            f"mode={getattr(decision, 'mode', None)}, top_k={eff_top_k}"
        )

        hits: List[RetrievalHit] = []
        if self.retriever is None:
            logger.warning("[RAG] retriever is not configured; returning empty evidence")
        else:
            if getattr(decision, "mode", None) == RoutingMode.GRAPH_AUGMENTED:
                hits = self._graph_augmented_retrieval(question, decision, eff_top_k)
            else:
                hits = self.retriever.search(question, top_k=eff_top_k)
                for h in hits:
                    h.source = "retriever"

        messages = self._build_messages(
            question=question,
            hits=hits,
            query_type=query_type_str,
        )
        msg_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            "[RAG] calling LLM.chat (provider=%s, model=%s), messages_chars=%d",
            self.cfg.llm.provider,
            self.cfg.llm.model,
            msg_chars,
        )

        llm = llm_override or self.llm

        raw_answer: str
        # if chat(messages=...) not supported, fallback to string prompt.
        try:
            raw_answer = llm.chat(messages=messages)
        except TypeError: 
            prompt = self._messages_to_prompt(messages)
            raw_answer = llm.chat(prompt=prompt)

        logger.info(
            "[RAG] LLM.chat returned type=%s, len=%s",
            type(raw_answer).__name__,
            len(raw_answer) if isinstance(raw_answer, str) else "NA",
        )
         
        raw_answer = _trim_to_answer(raw_answer)

        return RagAnswer(question=question, answer=raw_answer, hits=hits)
