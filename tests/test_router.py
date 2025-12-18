import pytest

from legalrag.routing.router import QueryRouter
from legalrag.schemas import QueryType, RoutingMode


class DummyLLM:
    """Simple fake LLM client used to test LLM fallback behavior."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.answer


def test_definition_query_uses_graph_augmented_mode():
    """包含“定义/是什么”一类关键词时，应路由到 DEFINITION + GRAPH_AUGMENTED。"""
    router = QueryRouter()
    question = "请解释一下合同解除的定义是什么？"

    decision = router.route(question)

    assert decision.query_type == QueryType.DEFINITION
    assert decision.mode == RoutingMode.GRAPH_AUGMENTED
    # 对应规则权重 weight=0.8
    assert decision.top_k_factor == pytest.approx(0.8)
    assert "DEFINITION" in decision.explain


def test_liability_query_routes_to_rag_with_higher_topk():
    """违约金/赔偿类问题，应路由到 LIABILITY + RAG，并放大 top_k。"""
    router = QueryRouter()
    question = "合同约定的违约金过高，如何调整赔偿责任？"

    decision = router.route(question)

    assert decision.query_type == QueryType.LIABILITY
    assert decision.mode == RoutingMode.RAG
    # 对应规则权重 weight=1.2
    assert decision.top_k_factor == pytest.approx(1.2)
    assert "LIABILITY" in decision.explain


def test_validity_query_routes_to_validity_rag():
    """包含“效力/无效/可撤销”等关键词时，应分类为 VALIDITY。"""
    router = QueryRouter()
    question = "格式条款中排除主要责任的条款效力如何认定？"

    decision = router.route(question)

    assert decision.query_type == QueryType.VALIDITY
    assert decision.mode == RoutingMode.RAG
    assert decision.top_k_factor == pytest.approx(1.2)
    assert "VALIDITY" in decision.explain


def test_termination_query_routes_to_termination_rag():
    """解除/终止类问题，应分类为 TERMINATION。"""
    router = QueryRouter()
    question = "一方迟延履行，另一方能否解除合同终止合作？"

    decision = router.route(question)

    assert decision.query_type == QueryType.TERMINATION
    assert decision.mode == RoutingMode.RAG
    assert decision.top_k_factor == pytest.approx(1.2)
    assert "TERMINATION" in decision.explain or "TERMINATION" in decision.explain.upper()


def test_normalization_does_not_break_routing():
    """问题中存在多余空白/换行时，应仍能正确匹配规则。"""
    router = QueryRouter()
    question = "  什么   是   合同   的   定义 ？  \n"

    decision = router.route(question)

    assert decision.query_type == QueryType.DEFINITION
    assert decision.mode == RoutingMode.GRAPH_AUGMENTED


def test_low_score_without_llm_falls_back_to_other():
    """无明显关键词、且未开启 LLM 时，应 fallback 到 OTHER + 默认 RAG。"""
    router = QueryRouter(llm_based=False)
    question = "最近合同纠纷很多，该怎么办？"

    decision = router.route(question)

    assert decision.query_type == QueryType.OTHER
    assert decision.mode == RoutingMode.RAG
    assert decision.top_k_factor == pytest.approx(1.0)
    assert "fallback to default OTHER" in decision.explain


def test_low_score_with_llm_uses_llm_fallback():
    """无明显规则匹配，且开启 llm_based 时，应调用 LLM 并按其结果路由。"""
    dummy_llm = DummyLLM(answer="DEFINITION")
    router = QueryRouter(llm_client=dummy_llm, llm_based=True)
    question = "请说明要约邀请的概念。"

    decision = router.route(question)

    # 确认确实调用了 LLM
    assert len(dummy_llm.calls) == 1
    # LLM 返回 DEFINITION → DEFINITION + GRAPH_AUGMENTED
    assert decision.query_type == QueryType.DEFINITION
    assert decision.mode == RoutingMode.GRAPH_AUGMENTED
    assert decision.top_k_factor == pytest.approx(1.0)
    assert "LLM fallback activated." in decision.explain


def test_high_score_does_not_trigger_llm_even_if_enabled():
    """规则匹配分数足够时，即使开启 llm_based 也不应触发 LLM fallback。"""
    dummy_llm = DummyLLM(answer="OTHER")
    router = QueryRouter(llm_client=dummy_llm, llm_based=True)
    question = "违约金如何调整？"

    decision = router.route(question)

    # 未触发 LLM
    assert len(dummy_llm.calls) == 0
    # 仍按规则路由
    assert decision.query_type == QueryType.LIABILITY
    assert decision.mode == RoutingMode.RAG
    assert decision.top_k_factor == pytest.approx(1.2)
