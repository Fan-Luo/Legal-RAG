from __future__ import annotations

from typing import List

from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.models import RagAnswer, RetrievalHit
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.retrieval.graph_store import LawGraphStore
from legalrag.routing.router import QueryRouter
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class RagPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.retriever = HybridRetriever(cfg)
        self.llm = LLMClient.from_config(cfg)
        self.graph = LawGraphStore(cfg)
        self.router = QueryRouter(
            llm_client=self.llm,
            llm_based=cfg.routing.llm_based,
        )

    def _build_prompt(self, question: str, hits: List[RetrievalHit]) -> str:
        ctx_lines = []
        for h in hits:
            c = h.chunk
            header = f"{c.law_name} {c.article_no}"
            if c.chapter:
                header = f"{c.chapter} - {header}"
            if c.section:
                header = f"{c.section} - {header}"
            ctx_lines.append(f"【条文 {h.rank} | {header}】\n{c.text}\n")
        context = "\n\n".join(ctx_lines)

        prompt = f"""你是一名精通中国《民法典·合同编》的法律助手，请严格依据给定的法律条文回答用户问题。

回答要求：
1. 先用自然语言给出结论和理由。
2. 明确引用具体的法律条文（篇、章、节、条号），格式类似：
   - 《民法典·合同编》第XXX条：……
3. 如果检索结果不足或无法判断，请明确说明“不足以作出明确判断”，不要编造法律条文。

【用户问题】
{question}

【检索到的候选法律条文（可能不完全相关）】
{context}

请根据上述条文给出专业、审慎的回答：
"""
        return prompt

    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        decision = self.router.route(question)
        eff_top_k = int((top_k or self.cfg.retrieval.top_k) * decision.top_k_factor)
        eff_top_k = max(3, min(eff_top_k, 30))

        logger.info(f"[RAG] query: {question}; mode={decision.mode}, top_k={eff_top_k}")

        hits = self.retriever.search(question, top_k=eff_top_k)

        prompt = self._build_prompt(question, hits)
        raw_answer = self.llm.chat(prompt)
        return RagAnswer(question=question, answer=raw_answer, hits=hits)
