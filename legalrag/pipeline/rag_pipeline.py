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

    def _build_prompt(self, question: str, hits: list[RetrievalHit]) -> str:

        ctx_lines = []
        for h in hits:
            c = h.chunk
            # Include chapter/section/article info
            header = f"{c.law_name} {c.article_no}"
            if getattr(c, "chapter", None):
                header = f"{c.chapter} - {header}"
            if getattr(c, "section", None):
                header = f"{c.section} - {header}"
            ctx_lines.append(f"【条文 {h.rank} | {header}】\n{c.text}\n")
        context = "\n\n".join(ctx_lines)

        prompt = f"""
你是一名精通中国《民法典·合同编》的法律助手，专门为用户提供基于现有法律条文的专业法律意见。请严格遵循以下规则：

【基本要求】
1. 先用自然语言给出结论和理由，逻辑清晰、条理分明。
2. 必须明确引用具体法律条文（篇、章、节、条号），格式示例如：
   - 《民法典·合同编》第XXX条：……
3. 仅依据提供的候选法律条文作答，不得自行编造或推测不存在的条文。
4. 如果现有法律条文不足以作出明确判断，应明确说明：
   - “根据提供条文，不足以作出明确判断。”
5. 回答尽量简明扼要，同时兼顾专业性和审慎性。
6. 严格遵循用户问题和上下文，不偏离主题。
7. 可以使用条目编号或分段形式，使答案更易阅读。

【输出结构化建议】
- 结论（Summary / Key Point）
- 法律依据（Legal Basis）  
- 说明/理由（Explanation / Reasoning）  

【用户问题】
{question}

【检索到的候选法律条文（可能不完全相关）】
{context}

【示例回答】

示例 1：
用户问题：甲公司未按合同约定支付货款，乙公司是否可以解除合同？
候选条文：
- 《民法典·合同编》第五百六十条：当事人一方不履行合同义务或者履行合同义务不符合约定的，对方可以要求履行或者采取补救措施。
- 《民法典·合同编》第五百七十条：当事人一方严重违反合同义务的，对方可以解除合同。

回答：
结论：乙公司可以解除合同。
法律依据：
- 《民法典·合同编》第五百七十条：当事人一方严重违反合同义务的，对方可以解除合同。
说明：甲公司未按合同约定支付货款，属于严重违反合同义务，依据第五百七十条，乙公司有权解除合同。

示例 2：
用户问题：丙方与丁方签订的合同条款存在歧义，该合同是否无效？
候选条文：
- 《民法典·合同编》第五十二条：依法成立的合同受法律保护。
- 《民法典·合同编》第五十四条：违反法律、行政法规的合同无效。

回答：
结论：不足以作出明确判断。
法律依据：根据提供条文，无明确条文说明合同条款歧义是否导致合同无效。
说明：现有条文不足以判断条款歧义的效力，无法确定合同是否无效。

【请根据上述条文和示例格式给出专业、审慎的回答】
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


    async def answer_async(self, question: str, top_k: int | None = None) -> RagAnswer:
        decision = self.router.route(question)
        eff_top_k = int((top_k or self.cfg.retrieval.top_k) * decision.top_k_factor)
        eff_top_k = max(3, min(eff_top_k, 30))

        logger.info(f"[RAG] query: {question}; mode={decision.mode}, top_k={eff_top_k}")

        # 检索条文
        hits = await asyncio.to_thread(self.retriever.search, question, eff_top_k)

        # 构建 Prompt
        prompt = self._build_prompt(question, hits)

        # 调用 LLM（同步或降级模式）
        if hasattr(self.llm, "chat_async"):
            raw_answer = await self.llm.chat_async(prompt)
        else:
            raw_answer = await asyncio.to_thread(self.llm.chat, prompt)

        return RagAnswer(question=question, answer=raw_answer, hits=hits)