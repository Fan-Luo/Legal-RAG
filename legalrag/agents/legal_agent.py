"""Lightweight Legal Agent wrapping RAG pipelines."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline
from legalrag.pipeline.multistep_pipeline import MultiStepRagPipeline
from legalrag.retrieval.case_retriever import CaseRetriever
from legalrag.llm.client import LLMClient
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class LegalAgent:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.simple_rag = RagPipeline(cfg)
        self.cot_rag = MultiStepRagPipeline(cfg)
        self.case_retriever = CaseRetriever(cfg)
        self.llm = LLMClient.from_config(cfg)

    def _route_rule_based(self, question: str) -> Dict[str, float]:
        q = (question or "").strip()
        if not q:
            return {"statute": 0.0, "case": 0.0}
        s = q.lower()

        statute_signals = [
            "条文", "法条", "法律规定", "依据", "适用", "构成要件", "要件",
            "解释", "如何理解", "如何适用", "条款", "article", "statute",
        ]
        case_signals = [
            "案例", "判例", "判决", "裁判", "法院观点", "类似案件",
            "怎么判", "判决理由", "裁判理由", "case", "precedent",
        ]

        statute_score = sum(1 for k in statute_signals if k in s)
        case_score = sum(1 for k in case_signals if k in s)

        # If question contains explicit fact pattern without legal reference, lean case.
        if any(k in s for k in ["事实", "经过", "情形", "起诉", "审理"]):
            case_score += 1
        return {"statute": float(statute_score), "case": float(case_score)}

    def _route_llm(self, question: str) -> Optional[Literal["statute", "case", "both"]]:
        if not self.cfg.routing.llm_based:
            return None
        prompt = (
            "Decide retrieval scope for the user question. "
            "Return only one token: statute, case, or both.\n\n"
            f"Question: {question}\nAnswer:"
        )
        try:
            out = (self.llm.chat(prompt=prompt, tag="agent_decide") or "").strip().lower()
        except Exception:
            return None
        if "both" in out:
            return "both"
        if "case" in out:
            return "case"
        if "statute" in out:
            return "statute"
        return None

    def decide_retrieval(self, question: str) -> Literal["statute", "case", "both"]:
        scores = self._route_rule_based(question)
        statute_score = scores["statute"]
        case_score = scores["case"]

        if statute_score == 0 and case_score == 0:
            llm_choice = self._route_llm(question)
            return llm_choice or "statute"

        if statute_score > 0 and case_score > 0:
            return "both"
        if case_score > statute_score:
            return "case"
        return "statute"

    def decompose(self, question: str) -> List[str]:
        """Very simple split for multi-part questions."""
        if "并且" in question:
            parts = [p.strip() for p in question.split("并且") if p.strip()]
            return parts
        if "以及" in question:
            parts = [p.strip() for p in question.split("以及") if p.strip()]
            return parts
        if " and " in question:
            parts = [p.strip() for p in question.split(" and ") if p.strip()]
            return parts
        return [question]

    def answer_complex(self, question: str) -> dict:
        subqs = self.decompose(question)
        logger.info("[Agent] decomposed into %d sub-questions", len(subqs))
        sub_answers = []
        for q in subqs:
            ans = self.cot_rag.answer(q)
            sub_answers.append({"question": q, "answer": ans.answer, "hits": ans.hits})

        combined_answer = "\n\n".join(
            [f"Sub-question: {sa['question']}\n{sa['answer']}" for sa in sub_answers]
        )

        return {
            "question": question,
            "sub_questions": subqs,
            "combined_answer": combined_answer,
        }

    def answer(self, question: str) -> dict:
        choice = self.decide_retrieval(question)
        logger.info("[Agent] retrieval_choice=%s", choice)
        out: Dict[str, object] = {"question": question, "retrieval": choice}

        if choice in {"statute", "both"}:
            out["statute"] = self.simple_rag.answer(question)
        if choice in {"case", "both"}:
            out["cases"] = self.case_retriever.search(question, top_k=5)
        return out
