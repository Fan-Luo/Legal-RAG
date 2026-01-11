from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from legalrag.config import AppConfig  
from legalrag.llm.client import LLMClient  
from legalrag.routing.legal_issue_extractor import (  
    LegalIssueExtractor,
    LegalIssueResult,
    LegalIssueType,
)


class RoutingMode(str, Enum):
    """
    Downstream code checks for suffix 'GRAPH_AUGMENTED', so keep that exact token.
    """
    RAG = "RAG"
    GRAPH_AUGMENTED = "GRAPH_AUGMENTED"


class QueryType(str, Enum):
    """
    Used mainly for prompt conditioning. Values are stable strings.
    """
    CONTRACT = "contract"
    TORT = "tort"
    PROPERTY = "property"
    FAMILY = "family"
    INHERITANCE = "inheritance"
    PERSONALITY = "personality"
    UNJUST_ENRICHMENT = "unjust_enrichment"
    AGENCY = "agency"
    GUARANTEE = "guarantee"
    GENERAL_CIVIL = "general_civil"
    OTHER = "other"


class RoutingDecision(BaseModel):
    mode: RoutingMode = RoutingMode.RAG
    query_type: QueryType = QueryType.GENERAL_CIVIL
    top_k_factor: float = 1.0
    explain: str = ""

    legal_issue_type: LegalIssueType = LegalIssueType.OTHER
    legal_issue_tags: List[str] = Field(default_factory=list)
    signals: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class QueryRouter:
    llm_client: Optional[LLMClient] = None
    llm_based: bool = False
    cfg: Optional[AppConfig] = None

    def __post_init__(self) -> None:
        self.issue_extractor = LegalIssueExtractor(llm=self.llm_client, cfg=self.cfg)

    def route(self, question: str) -> RoutingDecision:
        q = (question or "").strip()
        if not q:
            return RoutingDecision(
                mode=RoutingMode.RAG,
                query_type=QueryType.OTHER,
                top_k_factor=1.0,
                explain="empty question",
                legal_issue_type=LegalIssueType.OTHER,
                legal_issue_tags=[],
                signals={"empty": True},
            )

        issue: LegalIssueResult = self.issue_extractor.extract(q)
        mode = self._decide_mode(q, issue)
        qtype = self._map_issue_to_query_type(issue.legal_issue_type, q)
        top_k_factor = self._top_k_factor(q, issue)

        explain = "; ".join(
            [s for s in [issue.explain, f"mode={mode}", f"query_type={qtype}"] if s]
        )

        # LLM overrides the final decision
        if self.llm_based and self.llm_client is not None:
            try:
                mode, qtype, top_k_factor, llm_explain = self._llm_route(
                    q, issue, fallback=(mode, qtype, top_k_factor)
                )
                explain = "; ".join([s for s in [explain, llm_explain] if s])
            except Exception as e:  
                explain = "; ".join(
                    [s for s in [explain, f"llm_route_failed={type(e).__name__}"] if s]
                )

        return RoutingDecision(
            mode=mode,
            query_type=qtype,
            top_k_factor=float(top_k_factor),
            explain=explain,
            legal_issue_type=issue.legal_issue_type,
            legal_issue_tags=list(issue.tags or []),
            signals=issue.signals or {},
        )

    # -----------------------
    # Rule-based decisions
    # -----------------------
    def _decide_mode(self, q: str, issue: LegalIssueResult) -> RoutingMode:
        s = q.lower()
        has_article_marker = bool(issue.signals.get("has_article_ref"))
        interpretive = any(
            k in s
            for k in [
                "如何理解",
                "解释",
                "适用",
                "构成要件",
                "要件",
                "定义",
                "what is",
                "interpret",
                "meaning of",
                "article",
            ]
        )
        if has_article_marker or interpretive:
            return RoutingMode.GRAPH_AUGMENTED
        return RoutingMode.RAG

    def _map_issue_to_query_type(self, t: LegalIssueType, q: str) -> QueryType:
        if t in (
            LegalIssueType.CONTRACT_FORMATION,
            LegalIssueType.CONTRACT_BREACH,
            LegalIssueType.CONTRACT_REMEDIES,
            LegalIssueType.CONTRACT_INTERPRETATION,
        ):
            return QueryType.CONTRACT
        if t in (
            LegalIssueType.TORT_LIABILITY,
            LegalIssueType.PERSONAL_INJURY,
            LegalIssueType.PRODUCT_LIABILITY,
            LegalIssueType.PRIVACY_DEFAMATION,
        ):
            return QueryType.TORT
        if t in (
            LegalIssueType.PROPERTY_OWNERSHIP,
            LegalIssueType.REAL_ESTATE,
            LegalIssueType.MORTGAGE,
            LegalIssueType.POSSESSION,
        ):
            return QueryType.PROPERTY
        if t in (
            LegalIssueType.MARRIAGE_FAMILY,
            LegalIssueType.DIVORCE,
            LegalIssueType.CUSTODY,
            LegalIssueType.MAINTENANCE,
        ):
            return QueryType.FAMILY
        if t in (LegalIssueType.INHERITANCE, LegalIssueType.WILL_SUCCESSION):
            return QueryType.INHERITANCE
        if t in (LegalIssueType.PERSONALITY_RIGHTS,):
            return QueryType.PERSONALITY
        if t in (LegalIssueType.UNJUST_ENRICHMENT,):
            return QueryType.UNJUST_ENRICHMENT
        if t in (LegalIssueType.AGENCY,):
            return QueryType.AGENCY
        if t in (LegalIssueType.SURETY_GUARANTEE,):
            return QueryType.GUARANTEE
        return QueryType.GENERAL_CIVIL

    def _top_k_factor(self, q: str, issue: LegalIssueResult) -> float:
        s = q.lower()
        broad = any(
            k in s
            for k in [
                "有哪些",
                "如何",
                "怎么办",
                "what are",
                "how to",
                "can i",
                "should i",
                "是否可以",
            ]
        )
        has_article_marker = bool(issue.signals.get("has_article_ref"))
        if broad and not has_article_marker:
            return 1.35
        return 1.0

    # -----------------------
    # LLM-based routing
    # -----------------------
    def _llm_route(
        self,
        question: str,
        issue: LegalIssueResult,
        fallback: Tuple[RoutingMode, QueryType, float],
    ) -> Tuple[RoutingMode, QueryType, float, str]:
        mode0, qtype0, k0 = fallback
        llm = self.llm_client
        if llm is None:
            return mode0, qtype0, k0, "llm_route_skipped(no_llm)"

        sys = (
            "You are a routing module for a legal retrieval system. "
            "Return ONLY a JSON object with keys: mode, query_type, top_k_factor. "
            "mode in ['RAG','GRAPH_AUGMENTED']. "
            "query_type in ['contract','tort','property','family','inheritance','personality','unjust_enrichment','agency','guarantee','general_civil']. "
            "top_k_factor is a float in [0.8, 2.0]."
        )
        user = {
            "question": question,
            "heuristic_issue_type": str(issue.legal_issue_type),
            "heuristic_tags": issue.tags,
        }

        text = str(
            llm.chat(
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": str(user)},
                ]
            )
        )

        mode, qtype, k = mode0, qtype0, k0

        try:
            import json

            obj = json.loads(_extract_json(text))
            m = str(obj.get("mode", "")).strip().upper()
            qt = str(obj.get("query_type", "")).strip().lower()
            kk = float(obj.get("top_k_factor", k0))

            if m in ("RAG", "GRAPH_AUGMENTED"):
                mode = RoutingMode(m)
            if qt in [e.value for e in QueryType]:
                qtype = QueryType(qt)
            k = min(2.0, max(0.8, float(kk)))
            return mode, qtype, k, "llm_route_ok"
        except Exception:
            return mode0, qtype0, k0, "llm_route_parse_failed"


def _extract_json(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return "{}"
