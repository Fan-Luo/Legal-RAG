from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.routing.legal_issue_extractor import LegalIssueExtractor, IssueResult
from legalrag.schemas import IssueType, RoutingMode, TaskType
import json

class RoutingDecision(BaseModel):
    mode: RoutingMode = RoutingMode.RAG
    task_type: TaskType = TaskType.JUDGE_STYLE
    issue_type: IssueType = IssueType.OTHER
    top_k_factor: float = 1.0
    explain: str = ""

    issue_tags: List[str] = Field(default_factory=list)
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
                task_type=TaskType.JUDGE_STYLE,
                issue_type=IssueType.OTHER,
                top_k_factor=1.0,
                explain="empty question",
                issue_tags=[],
                signals={"empty": True},
            )

        issue: IssueResult = self.issue_extractor.extract(q)
        issue_type = issue.issue_type
        mode = None
        task_type = None
        top_k_factor = None
        explain_parts: List[str] = [issue.explain] if issue.explain else []

        def _rule_based() -> Tuple[RoutingMode, TaskType, IssueType, float]:
            return (
                self._decide_mode(q, issue),
                self._decide_task_type(q),
                issue.issue_type,
                self._top_k_factor(q, issue),
            )

        # Prefer LLM routing when available; fall back to rule-based only if LLM is unavailable or fails.
        if self.llm_based and self.llm_client is not None:
            try:
                mode, task_type, issue_type, top_k_factor, llm_explain = self._llm_route(
                    q, issue
                )
                if llm_explain:
                    explain_parts.append(llm_explain)
            except Exception as e:
                mode, task_type, issue_type, top_k_factor = _rule_based()
                explain_parts.append(f"llm_route_failed={type(e).__name__}")
        else:
            mode, task_type, issue_type, top_k_factor = _rule_based()

        explain_parts.extend([f"mode={mode}", f"task_type={task_type}", f"issue_type={issue_type}"])
        explain = "; ".join([s for s in explain_parts if s])

        return RoutingDecision(
            mode=mode,
            task_type=task_type,
            issue_type=issue_type,
            top_k_factor=float(top_k_factor),
            explain=explain,
            issue_tags=list(issue.tags or []),
            signals=issue.signals or {},
        )

    # -----------------------
    # Rule-based decisions
    # -----------------------
    def _decide_mode(self, q: str, issue: IssueResult) -> RoutingMode:
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

    def _decide_task_type(self, q: str) -> TaskType:
        s = q.lower()
        elements_patterns = [
            "构成要件",
            "成立要件",
            "构成要素",
            "要件有哪些",
            "要件是什么",
            "要素有哪些",
            "要素是什么",
            "需要哪些条件",
            "需要什么条件",
            "需要哪些要件",
            "需要什么要件",
            "适用前提",
            "适用条件",
            "前提是什么",
            "前提条件",
            "条件是什么",
            "条件有哪些",
            "elements of",
            "elements for",
            "requirements for",
            "prerequisites for",
            "conditions for",
            "what are the elements",
            "what are the requirements",
            "what are the conditions",
        ]
        if any(k in s for k in elements_patterns):
            return TaskType.ELEMENTS_CHECKLIST
        if any(k in s for k in ["是否可以", "能否", "可以", "能不能", "是否能", "can i", "can we", "is it possible"]):
            return TaskType.JUDGE_STYLE
        if any(k in s for k in ["什么是", "定义", "含义", "如何理解", "本法所称", "本条所称", "interpret", "meaning of"]):
            return TaskType.STATUTE_EXEGESIS
        if any(k in s for k in ["风险", "风险点", "注意事项", "提示", "risk", "alert"]):
            return TaskType.RISK_ALERT
        if any(k in s for k in ["区别", "对比", "比较", "差异", "versus", "compare"]):
            return TaskType.COMPARATIVE_RULES
        if any(k in s for k in ["证据", "举证", "证明", "程序", "流程", "起诉", "立案", "evidence", "procedure"]):
            return TaskType.PROCEDURE_EVIDENCE_LIST
        return TaskType.JUDGE_STYLE

    def _top_k_factor(self, q: str, issue: IssueResult) -> float:
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
        issue: IssueResult,
    ) -> Tuple[RoutingMode, TaskType, IssueType, float, str]:

        llm = self.llm_client

        sys = (
            "You are an intent classification component for a legal question answering system."
            "Your task is to select the most appropriate task type based on the user's question."
            "Return ONLY a JSON object with keys: mode, task_type, issue_type, top_k_factor. "
            "mode in ['RAG','GRAPH_AUGMENTED']. "
            f"task_type in {[e.value for e in TaskType]}. "
            "task_type meanings: "
            "judge_style: use when the user requests an overall determination (a conclusion-first answer such as permitted/not permitted, valid/invalid, liable/not liable) and supporting reasoning based on cited provisions. "
            "statute_exegesis: define/explain provisions without reaching a case-specific determination. "
            "risk_alert: identify legal/compliance risks, triggers, and missing facts. "
            "elements_checklist: use ONLY when the user explicitly requests an enumeration/checklist of required elements/conditions/prerequisites as the primary output (i.e., a condition list). "
            "Do NOT choose elements_checklist merely because a determination could be derived from conditions; unless a checklist is explicitly requested, choose judge_style. "
            "Tie-breaker: if ambiguous between judge_style and elements_checklist, choose judge_style. "
            "comparative_rules: compare rules and differences; "
            "procedure_evidence_list: procedure steps and evidence checklist; "
            "other: use only when none fit. "
            f"issue_type in {[e.value for e in IssueType]}. "
            "top_k_factor is a float in [0.8, 2.0]."
        )

        user = {
            "question": question,
            "heuristic_issue_type": str(issue.issue_type),
            "heuristic_tags": issue.tags,
        }

        text = str(
            llm.chat(
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": str(user)},
                ],
                tag="route_llm",
            )
        )

        try:
            obj = json.loads(_extract_json(text))
            m = str(obj.get("mode", "")).strip().upper()
            tt = str(obj.get("task_type", "")).strip().lower()
            it = str(obj.get("issue_type", "")).strip().lower()
            kk = float(obj.get("top_k_factor", k0))

            if m in ("RAG", "GRAPH_AUGMENTED"):
                mode = RoutingMode(m)
            if tt in [e.value for e in TaskType]:
                task_type = TaskType(tt)
            if it in [e.value for e in IssueType]:
                issue_type = IssueType(it)
            k = min(2.0, max(0.8, float(kk)))
            return mode, task_type, issue_type, k, "llm_route_ok"
        except Exception:
            return _, _, _, _, "llm_route_parse_failed"


def _extract_json(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return "{}"
