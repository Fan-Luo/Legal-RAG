from __future__ import annotations
from typing import List, Tuple, Optional, Dict

from legalrag.models import QueryType, RoutingMode, RoutingDecision
from legalrag.utils.text import normalize_whitespace


class Rule:
    def __init__(self, keywords: List[str], qtype: QueryType, weight: float, mode: RoutingMode):
        self.keywords = keywords
        self.qtype = qtype
        self.weight = weight
        self.mode = mode


class QueryRouter:
    def __init__(self, llm_client=None, llm_based: bool = False, threshold: float = 0.6):
        self.llm_client = llm_client
        self.llm_based = llm_based
        self.threshold = threshold

        # Rule base (can be extended easily)
        self.rules = [
            Rule(["定义", "含义", "是什么", "如何理解"], QueryType.DEFINITION, 0.8, RoutingMode.GRAPH_AUGMENTED),
            Rule(["违约金", "赔偿", "损失", "责任承担"], QueryType.LIABILITY, 1.2, RoutingMode.RAG),
            Rule(["效力", "无效", "可撤销"], QueryType.VALIDITY, 1.2, RoutingMode.RAG),
            Rule(["解除合同", "终止", "终止权"], QueryType.TERMINATION, 1.2, RoutingMode.RAG),
        ]

    def _score_rules(self, question: str) -> Tuple[Optional[Rule], float, List[str]]:
        explanations = []
        best_rule = None
        best_score = 0.0

        for rule in self.rules:
            match_count = sum(1 for kw in rule.keywords if kw in question)
            score = match_count * rule.weight

            if match_count > 0:
                explanations.append(f"Matched {match_count} keywords → {rule.qtype.name} (score={score})")

            if score > best_score:
                best_score = score
                best_rule = rule

        return best_rule, best_score, explanations

    def _ask_llm(self, question: str) -> Optional[Rule]:
        """LLM fallback when rule score is low or ambiguous."""
        prompt = f"""
你是合同法的法律问题分类器，请将问题分类为以下类别之一：
- 定义类（DEFINITION）
- 责任类（LIABILITY）
- 效力类（VALIDITY）
- 解除终止类（TERMINATION）
- 其他（OTHER）

问题：{question}

只返回类别英文名。
"""
        answer = self.llm_client.complete(prompt)
        mapping = {
            "DEFINITION": QueryType.DEFINITION,
            "LIABILITY": QueryType.LIABILITY,
            "VALIDITY": QueryType.VALIDITY,
            "TERMINATION": QueryType.TERMINATION,
            "OTHER": QueryType.OTHER,
        }

        qtype = mapping.get(answer.strip().upper(), QueryType.OTHER)
        mode = RoutingMode.RAG if qtype != QueryType.DEFINITION else RoutingMode.GRAPH_AUGMENTED

        return Rule([], qtype, 1.0, mode)

    def route(self, question: str) -> RoutingDecision:
        q = normalize_whitespace(question)

        best_rule, score, explanations = self._score_rules(q)

        # === Case 1: Rules confident enough ===
        if best_rule and score >= self.threshold:
            return RoutingDecision(
                query_type=best_rule.qtype,
                mode=best_rule.mode,
                top_k_factor=best_rule.weight,
                explain="; ".join(explanations)
            )

        # === Case 2: Rule-based weak → LLM fallback ===
        if self.llm_based and self.llm_client:
            llm_rule = self._ask_llm(q)
            explanations.append("LLM fallback activated.")
            return RoutingDecision(
                query_type=llm_rule.qtype,
                mode=llm_rule.mode,
                top_k_factor=llm_rule.weight,
                explain="; ".join(explanations)
            )

        # === Case 3: Default fallback ===
        return RoutingDecision(
            query_type=QueryType.OTHER,
            mode=RoutingMode.RAG,
            top_k_factor=1.0,
            explain="Rule score low; fallback to default OTHER."
        )
