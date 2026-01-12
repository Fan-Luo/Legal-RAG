import pytest

from legalrag.routing.router import QueryRouter
from legalrag.schemas import IssueType, RoutingMode, TaskType


class DummyLLM:
    """Simple fake LLM client used to test LLM override behavior."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    def chat(self, messages=None, prompt=None):
        self.calls.append({"messages": messages, "prompt": prompt})
        return self.answer


def test_statute_exegesis_routes_to_graph_augmented():
    router = QueryRouter()
    question = "请解释一下合同解除的定义是什么？"

    decision = router.route(question)

    assert decision.task_type == TaskType.STATUTE_EXEGESIS
    assert decision.issue_type == IssueType.CONTRACT_TERMINATION
    assert decision.mode == RoutingMode.GRAPH_AUGMENTED


def test_penalty_liquidated_issue_type():
    router = QueryRouter()
    question = "合同约定的违约金过高，如何调整？"

    decision = router.route(question)

    assert decision.issue_type == IssueType.PENALTY_LIQUIDATED
    assert decision.mode == RoutingMode.RAG


def test_empty_question_falls_back():
    router = QueryRouter()
    decision = router.route("")

    assert decision.issue_type == IssueType.OTHER
    assert decision.task_type == TaskType.JUDGE_STYLE
    assert decision.mode == RoutingMode.RAG


def test_llm_override_routing():
    dummy_llm = DummyLLM(
        answer='{"mode":"RAG","task_type":"risk_alert","issue_type":"tort_liability","top_k_factor":1.5}'
    )
    router = QueryRouter(llm_client=dummy_llm, llm_based=True)
    question = "交通事故造成损害怎么办？"

    decision = router.route(question)

    assert len(dummy_llm.calls) == 1
    assert decision.task_type == TaskType.RISK_ALERT
    assert decision.issue_type == IssueType.TORT_LIABILITY
    assert decision.top_k_factor == pytest.approx(1.5)
