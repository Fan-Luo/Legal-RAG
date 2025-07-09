from __future__ import annotations

from typing import List, Tuple, Optional

from legalrag.models import QueryType, RoutingMode, RoutingDecision
from legalrag.utils.text import normalize_whitespace


class Rule:
    """
    A simple rule mapping keywords -> (QueryType, RoutingMode, top_k_factor).

    weight:
      - used both as scoring weight (confidence) and as top_k_factor (how many hits to retrieve)
    priority:
      - used only for tie-break (higher wins)
    """

    def __init__(
        self,
        name: str,
        keywords: List[str],
        qtype: QueryType,
        weight: float,
        mode: RoutingMode,
        priority: int = 0,
    ):
        self.name = name
        self.keywords = keywords
        self.qtype = qtype
        self.weight = weight
        self.mode = mode
        self.priority = priority


class QueryRouter:
    """
    Rule-first router for contract-law style questions.

    6 categories:
      - DEFINITION: 概念/定义
      - VALIDITY: 成立/生效/无效/可撤销
      - PERFORMANCE: 履行/抗辩/风险承担/价款
      - BREACH_REMEDY: 违约责任/违约金/赔偿/定金等救济
      - TERMINATION: 解除/终止/解除后果
      - PROCEDURE: 诉讼时效/举证/程序性问题

    Graph-mode suggestions:
      - DEFINITION: GRAPH_AUGMENTED (definitions + cross-ref heavy)
      - PROCEDURE: GRAPH_AUGMENTED (time limits, interruption/suspension cross-ref)
      - VALIDITY: GRAPH_AUGMENTED (voidable/invalid often cross-ref)
      - others: default RAG
    """

    def __init__(self, llm_client=None, llm_based: bool = False, threshold: float = 0.9):
        self.llm_client = llm_client
        self.llm_based = llm_based
        self.threshold = threshold

        # Notes:
        # - weight works as both "confidence weight" and "top_k_factor".
        # - threshold default 0.9: a single strong match (weight>=1.0) usually triggers.
        self.rules: List[Rule] = [
            # PROCEDURE (often should override, hence high priority)
            Rule(
                name="procedure",
                keywords=[
                    "诉讼时效", "时效", "起算", "中止", "中断", "延长", "届满",
                    "除斥期间", "期间", "催告", "通知到达", "送达",
                    "举证", "证明责任", "证据", "管辖", "仲裁", "起诉", "受理", "程序",
                    "是否及时", "是否超过", "两年", "三年", "一年", "知道或者应当知道",
                ],
                qtype=QueryType.PROCEDURE,
                weight=1.3,
                mode=RoutingMode.GRAPH_AUGMENTED,
                priority=60,
            ),

            # VALIDITY
            Rule(
                name="validity",
                keywords=[
                    "成立", "生效", "无效", "效力", "当然无效", "部分无效",
                    "可撤销", "撤销", "撤销权", "撤销期限",
                    "重大误解", "欺诈", "胁迫", "乘人之危", "显失公平",
                    "意思表示", "表见代理", "代理权限", "恶意串通",
                    "违反强制性规定", "违背公序良俗",
                ],
                qtype=QueryType.VALIDITY,
                weight=1.2,
                mode=RoutingMode.GRAPH_AUGMENTED,
                priority=50,
            ),

            # DEFINITION
            Rule(
                name="definition",
                keywords=[
                    "定义", "含义", "概念", "是什么", "什么是", "解释", "如何理解", "本法所称",
                    "何谓", "如何认定", "构成要件",
                ],
                qtype=QueryType.DEFINITION,
                weight=1.25,
                mode=RoutingMode.GRAPH_AUGMENTED,
                priority=40,
            ),

            # TERMINATION
            Rule(
                name="termination",
                keywords=[
                    "解除", "解除合同", "解除权", "法定解除", "约定解除",
                    "终止", "终止权", "解除通知", "解除期限", "解除后果",
                    "返还", "恢复原状", "结算", "损失", "继续履行不能",
                ],
                qtype=QueryType.TERMINATION,
                weight=1.2,
                mode=RoutingMode.RAG,
                priority=30,
            ),

            # BREACH_REMEDY
            Rule(
                name="breach_remedy",
                keywords=[
                    "违约", "违约责任", "违约金", "赔偿", "损失赔偿", "损害赔偿",
                    "定金", "双倍返还", "继续履行", "采取补救措施", "减少价款",
                    "违约方", "守约方", "责任承担", "赔偿范围", "可得利益",
                    "与损失赔偿能否并存", "能否并存",
                ],
                qtype=QueryType.BREACH_REMEDY,
                weight=1.25,
                mode=RoutingMode.RAG,
                priority=20,
            ),

            # PERFORMANCE
            Rule(
                name="performance",
                keywords=[
                    "履行", "履行期限", "履行地点", "履行方式",
                    "交付", "受领", "价款", "付款", "支付", "结算",
                    "质量", "数量", "标的", "瑕疵", "检验", "修理", "更换", "重作",
                    "风险承担", "风险转移",
                    "同时履行抗辩", "先履行抗辩", "不安抗辩", "抗辩",
                    "代位", "撤销权",  # 有时也跟履行/保全相关（若你后续加“债的保全”可细分）
                ],
                qtype=QueryType.PERFORMANCE,
                weight=1.15,
                mode=RoutingMode.RAG,
                priority=10,
            ),
        ]

    def _score_rules(self, question: str) -> Tuple[Optional[Rule], float, List[str]]:
        explanations: List[str] = []
        best_rule: Optional[Rule] = None
        best_score: float = 0.0

        for rule in self.rules:
            matched = [kw for kw in rule.keywords if kw in question]
            match_count = len(matched)
            if match_count <= 0:
                continue

            # score: count * weight; tie-break by priority
            score = match_count * rule.weight

            explanations.append(
                f"[{rule.name}] matched={match_count} weight={rule.weight:.2f} "
                f"score={score:.2f} qtype={rule.qtype.name} mode={rule.mode.value} "
                f"examples={matched[:4]}"
            )

            if (score > best_score) or (
                score == best_score and best_rule is not None and rule.priority > best_rule.priority
            ) or (best_rule is None):
                best_score = score
                best_rule = rule

        return best_rule, best_score, explanations

    def _ask_llm(self, question: str) -> Optional[Rule]:
        """
        LLM fallback when rule score is low or ambiguous.
        Expect EXACT one of:
          DEFINITION / VALIDITY / PERFORMANCE / BREACH_REMEDY / TERMINATION / PROCEDURE / OTHER
        """
        if self.llm_client is None:
            return None

        prompt = f"""
你是合同法法律问题分类器，请把问题分类为以下类别之一，并只返回英文类别名（不要解释）：

- DEFINITION（概念/定义）
- VALIDITY（成立/生效/无效/可撤销）
- PERFORMANCE（履行/抗辩/风险承担/价款）
- BREACH_REMEDY（违约责任/违约金/赔偿/定金等救济）
- TERMINATION（解除/终止/解除后果）
- PROCEDURE（诉讼时效/举证/程序性问题）
- OTHER（其他）

问题：{question}
""".strip()

        answer = self.llm_client.complete(prompt) if hasattr(self.llm_client, "complete") else self.llm_client.chat(prompt)
        label = (answer or "").strip().upper()

        mapping = {
            "DEFINITION": (QueryType.DEFINITION, RoutingMode.GRAPH_AUGMENTED, 1.1),
            "VALIDITY": (QueryType.VALIDITY, RoutingMode.GRAPH_AUGMENTED, 1.2),
            "PERFORMANCE": (QueryType.PERFORMANCE, RoutingMode.RAG, 1.15),
            "BREACH_REMEDY": (QueryType.BREACH_REMEDY, RoutingMode.RAG, 1.25),
            "TERMINATION": (QueryType.TERMINATION, RoutingMode.RAG, 1.2),
            "PROCEDURE": (QueryType.PROCEDURE, RoutingMode.GRAPH_AUGMENTED, 1.3),
            "OTHER": (QueryType.OTHER, RoutingMode.RAG, 1),
        }
        qtype, mode, weight = mapping.get(label, (QueryType.PERFORMANCE, RoutingMode.RAG, 1.0))
        return Rule(name="llm_fallback", keywords=[], qtype=qtype, weight=weight, mode=mode, priority=0)

    def route(self, question: str) -> RoutingDecision:
        q = normalize_whitespace(question)

        best_rule, score, explanations = self._score_rules(q)

        # Case 1: Rules confident enough
        if best_rule and score >= self.threshold:
            return RoutingDecision(
                query_type=best_rule.qtype,
                mode=best_rule.mode,
                top_k_factor=best_rule.weight,
                explain="; ".join(explanations) if explanations else f"rule={best_rule.name}",
            )

        # Case 2: Rule-based weak -> LLM fallback
        if self.llm_based and self.llm_client:
            llm_rule = self._ask_llm(q)
            if llm_rule:
                explanations.append(f"LLM fallback activated (score={score:.2f} < threshold={self.threshold:.2f}).")
                return RoutingDecision(
                    query_type=llm_rule.qtype,
                    mode=llm_rule.mode,
                    top_k_factor=llm_rule.weight,
                    explain="; ".join(explanations),
                )

        # Case 3: Default fallback
        return RoutingDecision(
            query_type=QueryType.OTHER,
            mode=RoutingMode.RAG,
            top_k_factor=1.0,
            explain="Rule score low; fallback to default OTHER."
        )
