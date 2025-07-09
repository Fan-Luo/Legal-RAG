from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.models import (
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
    """
    简单余弦相似度实现，a/b 都是一维向量。
    """
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def extract_key_term(question: str) -> str:
    """
    非严格的关键词抽取，用于 definition-special 行为。
    这里先做一个非常简单的版本：
    - 截掉末尾问号
    - 尝试去掉常见前缀：'什么是' / '如何理解' / '本法所称'
    """
    q = question.strip().rstrip("？?")

    patterns = ["什么是", "如何理解", "本法所称", "本条所称", "本编所称"]
    for p in patterns:
        if q.startswith(p):
            return q[len(p):].strip()

    return q


class RagPipeline:
    """
    Graph-aware RAG Pipeline（2025 工程版）

    - HybridRetriever: dense(BGE) + sparse(BM25)
    - LawGraphStore: 结构图 / 概念图
    - QueryRouter: QueryType + RoutingMode 决策
    - Graph-augmented retrieval:
        * 从 dense+sparse 结果获取种子条文
        * 在图上做多跳扩展
        * 对 definition 类问题做额外扩展
        * 用 BGE 再算一遍语义得分做 re-ranking
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # 基础组件
        self.retriever = HybridRetriever(cfg)
        self.llm = LLMClient.from_config(cfg)
        self.graph = LawGraphStore(cfg)
        self.router = QueryRouter(
            llm_client=self.llm,
            llm_based=cfg.routing.llm_based,
        )

        # 与 VectorStore 使用相同的 BGE encoder
        model_name = cfg.retrieval.embedding_model  # e.g. "BAAI/bge-base-zh-v1.5"
        logger.info(f"[RAG] Loading embedding model for graph-aware rerank: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embed_model.eval()
        self.prompt_path = Path("legalrag/prompts/legal_rag_prompt.txt")
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8")
    
    # Embedding helper 
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        使用 BGE 做句向量编码：
        - tokenizer → model → mean pooling
        - L2 归一化，方便用余弦相似度
        """
        if not texts:
            # 这里假设 hidden_size=768；如果要更严谨，可以使用 self.embed_model.config.hidden_size
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
        emb = emb / norms
        return emb


    # Graph-aware retrieval
    def _graph_augmented_retrieval(
        self,
        question: str,
        decision,
        top_k: int,
    ) -> List[RetrievalHit]:
        """
        Graph-aware 检索流程：
        1. baseline：HybridRetriever（BGE dense + BM25 sparse）
        2. 取前若干条文作为 seed，在法条图上做多跳扩展
        3. definition 类问题：根据关键词从定义节点扩展
        4. 把 baseline hits + graph hits 合并
        5. 用 BGE 重新计算语义相关度，做加权融合 & re-ranking
        6. 为每条命中生成 explain 字段：来源、语义得分、graph_depth/relations
        """
        #  Step 1: baseline RAG hits 
        rag_hits = self.retriever.search(question, top_k=top_k)
        for h in rag_hits:
            h.source = "retriever"

        seed_ids = [
            getattr(h.chunk, "id", None) or getattr(h.chunk, "article_id", None)
            for h in rag_hits
        ]
        seed_ids = [sid for sid in seed_ids if sid]

        #  Step 2: graph expansion 
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
                    neighbors = self.graph.get_neighbors(sid, depth=2)
                    graph_nodes.extend(neighbors)
                except TypeError:
                    neighbors = self.graph.get_neighbors(sid)
                    graph_nodes.extend(neighbors)

        #  Step 3: definition-special expansion 
        if decision.query_type == QueryType.DEFINITION and hasattr(self.graph, "get_definition_sources"):
            term = extract_key_term(question)
            try:
                def_nodes = self.graph.get_definition_sources(term)
                graph_nodes.extend(def_nodes)
            except Exception:
                logger.warning("[RAG][graph] get_definition_sources failed", exc_info=True)

        # 去重（避免和 baseline 结果重复）
        seen_ids = set(seed_ids)
        unique_nodes = []
        for n in graph_nodes:
            aid = getattr(n, "article_id", None) or getattr(n, "id", None)
            if not aid or aid in seen_ids:
                continue
            seen_ids.add(aid)
            unique_nodes.append(n)

        #  Step 4: convert graph nodes to RetrievalHit 
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
                    score=0.5,  # 初始 graph 得分，后续会融合语义得分
                    rank=-1,
                    source="graph",
                    graph_depth=getattr(n, "graph_depth", None),
                    relations=getattr(n, "relations", None),
                )
            )

        #  Step 5: semantic re-ranking（dense + sparse + graph 融合） 
        combined: List[RetrievalHit] = rag_hits + graph_hits

        if not combined:
            return []

        texts = [h.chunk.text or "" for h in combined]
        vecs = self._encode_texts(texts)
        qvec = self._encode_texts([question])[0]

        for h, v in zip(combined, vecs):
            sem = cosine_similarity(qvec, v)
            h.semantic_score = sem
            # 0.6 * 原始分数（dense+bm25 或 graph 初始分） + 0.4 * 语义分
            h.score = 0.6 * float(h.score) + 0.4 * sem

        # definition-special 的额外 boost
        if decision.query_type == QueryType.DEFINITION:
            combined = self._boost_definition_hits(combined)

        #  Step 6: final rank 
        combined.sort(key=lambda h: h.score, reverse=True)
        top = combined[:top_k]
        for i, h in enumerate(top, start=1):
            h.rank = i

        #  Step 7: explainability 
        for h in top:
            h.explain = {
                "source": getattr(h, "source", None),
                "semantic_score": getattr(h, "semantic_score", None),
                "graph_depth": getattr(h, "graph_depth", None),
                "relations": getattr(h, "relations", None),
            }

        return top

    def _boost_definition_hits(self, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """
        对“定义类”问题进行额外提升：
        - 文本中包含 “是指 / 本法所称 / 本条所称 / 本编所称” 等表达
        - 属于“总则”章节
        - graph 关系为 same_section / sibling
        """
        def is_definition_like(chunk: LawChunk) -> bool:
            text = (chunk.text or "").replace("\n", "")
            patterns = ["是指", "本法所称", "本条所称", "本编所称"]
            return any(p in text for p in patterns)

        boosted: List[RetrievalHit] = []
        for h in hits:
            bonus = 0.0
            if is_definition_like(h.chunk):
                bonus += 0.2
            if getattr(h.chunk, "chapter", None) and "总则" in h.chunk.chapter:
                bonus += 0.1
            # 如果 RetrievalHit 有 relation / relations 字段，可以做进一步加权
            rel = getattr(h, "relation", None)
            if rel in ("same_section", "sibling"):
                bonus += 0.05

            h.score = float(h.score) + bonus
            boosted.append(h)

        return boosted

    def _build_prompt(
        self,
        question: str,
        hits: List[RetrievalHit],
        decision: Optional[object] = None,
    ) -> str:
        """
        构造面向中文法律咨询场景的 RAG Prompt
        """
        law_context_parts = []
        for i, h in enumerate(hits, start=1):
            c = h.chunk
            article_no = getattr(c, "article_no", "") or ""
            chapter = getattr(c, "chapter", "") or ""
            section = getattr(c, "section", "") or ""
            text = (c.text or "").strip()
            score = getattr(h, "score", 0.0)
            source = getattr(h, "source", "")

            law_context_parts.append(
                f"[候选条文 {i}] （score={score:.4f}, source={source})\n"
                f"条号：{article_no}\n"
                f"章节：{chapter} {section}\n"
                f"内容：{text}\n"
            )

        law_context = "\n\n".join(law_context_parts) if law_context_parts else "（当前未检索到相关条文）"

        query_type_str = ""
        if decision is not None and hasattr(decision, "query_type"):
            query_type_str = f"（推测问题类型：{decision.query_type}）"

 

        return self.prompt_template.format(
            query_type=query_type_str,
            question=question,
            law_context=law_context,
        )

    # Public API
    def answer(self, question: str, top_k: Optional[int] = None) -> RagAnswer:
        """
        同步 RAG 入口：
        - 自动根据 Router 决策是否使用 Graph-augmented 模式
        - 返回 RagAnswer（包括回答文本和命中条文列表）
        """
        decision = self.router.route(question)
        base_k = top_k or self.cfg.retrieval.top_k
        eff_top_k = int(base_k * getattr(decision, "top_k_factor", 1.0))
        eff_top_k = max(3, min(eff_top_k, 30))

        logger.info(
            f"[RAG] query: {question}; query_type={getattr(decision, 'query_type', None)}, "
            f"mode={getattr(decision, 'mode', None)}, top_k={eff_top_k}"
        )

        if getattr(decision, "mode", None) == RoutingMode.GRAPH_AUGMENTED:
            hits = self._graph_augmented_retrieval(question, decision, eff_top_k)
        else:
            hits = self.retriever.search(question, top_k=eff_top_k)
            for h in hits:
                h.source = "retriever"

        prompt = self._build_prompt(question, hits, decision)
        logger.info("[RAG] calling LLM.chat (provider=%s, model=%s), prompt_chars=%d",
                         self.cfg.llm.provider, self.cfg.llm.model, len(prompt))

        raw_answer = self.llm.chat(prompt)

        logger.info("[RAG] LLM.chat returned type=%s, len=%s",
                         type(raw_answer).__name__, len(raw_answer) if isinstance(raw_answer, str) else "NA")

        if isinstance(raw_answer, dict):
            raw_answer = raw_answer.get("text") or raw_answer.get("answer") or str(raw_answer)
        raw_answer = str(raw_answer)

        return RagAnswer(
            question=question,
            answer=raw_answer,
            hits=hits,
        )
