from __future__ import annotations

from legalrag.models import QueryType, RoutingMode, RoutingDecision
from legalrag.utils.text import normalize_whitespace


class QueryRouter:
    def __init__(self, llm_client=None, llm_based: bool = False):
        self.llm_client = llm_client
        self.llm_based = llm_based

    def route(self, question: str) -> RoutingDecision:
        q = normalize_whitespace(question)
        if any(k in q for k in ["定义", "含义", "是什么", "如何理解"]):
            qtype = QueryType.DEFINITION
            mode = RoutingMode.GRAPH_AUGMENTED
            factor = 0.8
        elif any(k in q for k in ["违约金", "赔偿", "损失", "责任承担"]):
            qtype = QueryType.LIABILITY
            mode = RoutingMode.RAG
            factor = 1.2
        elif any(k in q for k in ["效力", "无效", "可撤销"]):
            qtype = QueryType.VALIDITY
            mode = RoutingMode.RAG
            factor = 1.2
        elif any(k in q for k in ["解除合同", "终止", "终止权"]):
            qtype = QueryType.TERMINATION
            mode = RoutingMode.RAG
            factor = 1.2
        else:
            qtype = QueryType.OTHER
            mode = RoutingMode.RAG
            factor = 1.0

        return RoutingDecision(query_type=qtype, mode=mode, top_k_factor=factor)
