"""
Issue extractor: lightweight civil-law issue classifier (CN + EN).

- Heuristic-first; no ML dependencies required.
- Optional LLM refinement via cfg.routing.issue_llm_refine (default False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from legalrag.schemas import IssueType


class IssueResult(BaseModel):
    issue_type: IssueType = IssueType.OTHER
    tags: List[str] = Field(default_factory=list)
    explain: str = ""
    signals: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class LegalIssueExtractor:
    llm: Optional[Any] = None
    cfg: Optional[Any] = None

    def extract(self, question: str) -> IssueResult:
        q = (question or "").strip()
        s = q.lower()

        tags: List[str] = []
        signals: Dict[str, Any] = {}

        has_article_ref = _has_article_ref(q)
        signals["has_article_ref"] = has_article_ref

        issue_type = _match_issue_type(s)
        if issue_type is IssueType.OTHER:
            issue_type = _match_part_type(s)

        part_tag = _issue_part_tag(issue_type)
        if part_tag:
            tags.append(f"part:{part_tag}")
        tags.append(f"issue:{issue_type.value}")
        if has_article_ref:
            tags.append("article_ref")

        explain = f"heuristic_issue_type={issue_type}"
        out = IssueResult(issue_type=issue_type, tags=tags, explain=explain, signals=signals)

        # Optional LLM refine  
        if self.llm is not None and bool(getattr(getattr(self.cfg, "routing", None), "issue_llm_refine", False)):
            try:
                out = self._llm_refine(question, out)
            except Exception:
                return out
        return out

    def _llm_refine(self, question: str, base: IssueResult) -> IssueResult:
        llm = self.llm
        if llm is None:
            return base

        sys = (
            "Classify the user question into a civil-law issue type. "
            "Return ONLY JSON with keys: issue_type, tags. "
            f"issue_type must be one of: {[e.value for e in IssueType]}."
        )
        user = {"question": question, "heuristic": base.model_dump()}
        text = str(llm.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": str(user)}], tag="issue_refine"))

        import json
        obj = json.loads(_extract_json(text))
        t = str(obj.get("issue_type", "")).strip()
        tags = obj.get("tags", []) or []
        if t in [e.value for e in IssueType]:
            base.issue_type = IssueType(t)
        if isinstance(tags, list):
            base.tags = [str(x) for x in tags if str(x)]
        base.explain = (base.explain + "; llm_refine_ok").strip("; ")
        return base


def _kw_score(text: str, kws: List[str]) -> int:
    return sum(1 for k in kws if k and k.lower() in text)


def _match_issue_type(text: str) -> IssueType:
    rules = [
        (IssueType.PENALTY_LIQUIDATED, ["违约金", "liquidated", "penalty"]),
        (IssueType.DEPOSIT, ["定金", "订金", "deposit", "earnest"]),
        (IssueType.CONTRACT_TERMINATION, ["解除", "终止", "rescission", "terminate", "termination"]),
        (IssueType.DEFECTIVE_PERFORMANCE, ["瑕疵", "不合格", "缺陷", "defective", "nonconforming"]),
        (IssueType.PERFORMANCE_DEFENSE, ["先履行", "同时履行", "不安抗辩", "抗辩", "defense of performance", "concurrent"]),
        (IssueType.CONTRACT_FORMATION, ["订立", "成立", "要约", "承诺", "formation", "offer", "acceptance"]),
        (IssueType.CONTRACT_VALIDITY, ["效力", "无效", "可撤销", "validity", "void", "voidable"]),
        (IssueType.CONTRACT_INTERPRETATION, ["解释", "条款", "理解", "term", "clause", "interpret"]),
        (IssueType.CONTRACT_PERFORMANCE, ["履行", "交付", "付款", "performance", "delivery"]),
        (IssueType.BREACH_REMEDY, ["违约", "赔偿", "损害", "damages", "breach", "remedy"]),
        (IssueType.CONTRACT_TRANSFER, ["变更", "转让", "让与", "assignment", "transfer", "novation"]),
        (IssueType.GUARANTEE, ["保证", "担保", "surety", "guarantee"]),
        (IssueType.NEGOTIORUM_GESTIO, ["无因管理", "negotiorum"]),
        (IssueType.UNJUST_ENRICHMENT, ["不当得利", "unjust enrichment"]),
        (IssueType.OWNERSHIP, ["所有权", "ownership"]),
        (IssueType.POSSESSION, ["占有", "possession"]),
        (IssueType.REGISTRATION, ["登记", "registration"]),
        (IssueType.NEIGHBOR_RELATION, ["相邻关系", "neighbor"]),
        (IssueType.PROPERTY_USE_RIGHT, ["用益物权", "建设用地", "宅基地", "居住权", "地役权", "usufruct"]),
        (IssueType.MORTGAGE, ["抵押", "mortgage"]),
        (IssueType.PLEDGE, ["质押", "pledge"]),
        (IssueType.LIEN, ["留置", "lien"]),
        (IssueType.CIVIL_CAPACITY, ["民事权利能力", "民事行为能力", "capacity"]),
        (IssueType.CIVIL_ACT_VALIDITY, ["民事法律行为", "意思表示", "行为效力", "legal act", "juridical act"]),
        (IssueType.AGENCY, ["代理", "委托", "授权", "表见代理", "agency", "power of attorney", "apparent authority"]),
        (IssueType.CIVIL_LIABILITY, ["民事责任", "责任形式", "liability"]),
        (IssueType.LIMITATION_PERIOD, ["诉讼时效", "时效", "limitation period"]),
        (IssueType.NAME_RIGHT, ["姓名权", "名称权", "name right"]),
        (IssueType.PORTRAIT_RIGHT, ["肖像权", "portrait"]),
        (IssueType.REPUTATION_RIGHT, ["名誉权", "reputation"]),
        (IssueType.PRIVACY_INFO, ["隐私", "个人信息", "privacy", "personal information"]),
        (IssueType.PERSONALITY_INFRINGEMENT, ["人格权", "肖像", "名誉", "隐私", "personality", "defamation"]),
        (IssueType.MARRIAGE, ["结婚", "婚姻", "marriage"]),
        (IssueType.DIVORCE, ["离婚", "divorce"]),
        (IssueType.FAMILY_PROPERTY, ["夫妻共同财产", "家庭财产", "marital property"]),
        (IssueType.CUSTODY_SUPPORT, ["抚养", "监护", "扶养", "赡养", "custody", "support"]),
        (IssueType.INHERITANCE_WILL, ["遗嘱", "will"]),
        (IssueType.INHERITANCE_STATUTORY, ["法定继承", "statutory succession"]),
        (IssueType.INHERITANCE_SHARE, ["继承份额", "继承顺序", "share", "order of succession"]),
        (IssueType.PERSONAL_INJURY, ["人身损害", "personal injury", "injury"]),
        (IssueType.PRODUCT_LIABILITY, ["产品责任", "缺陷产品", "product liability"]),
        (IssueType.MEDICAL_TORT, ["医疗损害", "medical"]),
        (IssueType.TORT_LIABILITY, ["侵权", "tort", "liability"]),
    ]
    for issue, kws in rules:
        if _kw_score(text, kws) > 0:
            return issue
    return IssueType.OTHER


def _match_part_type(text: str) -> IssueType:
    part_rules = {
        IssueType.CONTRACT: ["合同", "违约", "履行", "定金", "违约金", "解除", "合同条款", "contract", "breach", "performance"],
        IssueType.PROPERTY: ["物权", "所有权", "占有", "不动产", "动产", "登记", "抵押", "质押", "留置", "相邻关系", "用益物权", "property", "ownership"],
        IssueType.PERSONALITY: ["人格权", "名誉", "隐私", "肖像", "姓名权", "个人信息", "personality", "reputation", "privacy"],
        IssueType.MARRIAGE_FAMILY: ["婚姻", "结婚", "离婚", "夫妻", "抚养", "监护", "收养", "赡养", "marriage", "divorce", "custody"],
        IssueType.INHERITANCE: ["继承", "遗嘱", "遗产", "继承人", "法定继承", "inheritance", "will", "succession"],
        IssueType.TORT: ["侵权", "过错", "人身损害", "精神损害", "产品责任", "医疗损害", "tort", "liability", "injury"],
        IssueType.QUASI_CONTRACT: ["无因管理", "不当得利", "negotiorum", "unjust enrichment"],
        IssueType.GENERAL_CIVIL: ["民事", "自然人", "法人", "非法人组织", "民事权利", "意思表示", "代理", "民事责任", "诉讼时效", "期间", "capacity", "legal act"],
    }
    scores = {k: _kw_score(text, v) for k, v in part_rules.items()}
    top_issue, top_score = max(scores.items(), key=lambda x: float(x[1]))
    return top_issue if top_score > 0 else IssueType.OTHER


def _issue_part_tag(issue_type: IssueType) -> str:
    contract = {
        IssueType.CONTRACT,
        IssueType.CONTRACT_FORMATION,
        IssueType.CONTRACT_VALIDITY,
        IssueType.CONTRACT_INTERPRETATION,
        IssueType.CONTRACT_PERFORMANCE,
        IssueType.PERFORMANCE_DEFENSE,
        IssueType.DEFECTIVE_PERFORMANCE,
        IssueType.CONTRACT_TERMINATION,
        IssueType.BREACH_REMEDY,
        IssueType.PENALTY_LIQUIDATED,
        IssueType.DEPOSIT,
        IssueType.GUARANTEE,
        IssueType.CONTRACT_TRANSFER,
    }
    property_part = {
        IssueType.PROPERTY,
        IssueType.OWNERSHIP,
        IssueType.POSSESSION,
        IssueType.REGISTRATION,
        IssueType.NEIGHBOR_RELATION,
        IssueType.PROPERTY_USE_RIGHT,
        IssueType.MORTGAGE,
        IssueType.PLEDGE,
        IssueType.LIEN,
    }
    personality = {
        IssueType.PERSONALITY,
        IssueType.NAME_RIGHT,
        IssueType.PORTRAIT_RIGHT,
        IssueType.REPUTATION_RIGHT,
        IssueType.PRIVACY_INFO,
        IssueType.PERSONALITY_INFRINGEMENT,
    }
    marriage_family = {
        IssueType.MARRIAGE_FAMILY,
        IssueType.MARRIAGE,
        IssueType.DIVORCE,
        IssueType.FAMILY_PROPERTY,
        IssueType.CUSTODY_SUPPORT,
    }
    inheritance = {
        IssueType.INHERITANCE,
        IssueType.INHERITANCE_WILL,
        IssueType.INHERITANCE_STATUTORY,
        IssueType.INHERITANCE_SHARE,
    }
    tort = {
        IssueType.TORT,
        IssueType.TORT_LIABILITY,
        IssueType.PERSONAL_INJURY,
        IssueType.PRODUCT_LIABILITY,
        IssueType.MEDICAL_TORT,
    }
    general = {
        IssueType.GENERAL_CIVIL,
        IssueType.CIVIL_CAPACITY,
        IssueType.CIVIL_ACT_VALIDITY,
        IssueType.AGENCY,
        IssueType.CIVIL_LIABILITY,
        IssueType.LIMITATION_PERIOD,
    }
    quasi = {IssueType.QUASI_CONTRACT, IssueType.NEGOTIORUM_GESTIO, IssueType.UNJUST_ENRICHMENT}

    if issue_type in contract:
        return "contract"
    if issue_type in property_part:
        return "property"
    if issue_type in personality:
        return "personality"
    if issue_type in marriage_family:
        return "marriage_family"
    if issue_type in inheritance:
        return "inheritance"
    if issue_type in tort:
        return "tort"
    if issue_type in general:
        return "general"
    if issue_type in quasi:
        return "quasi_contract"
    return ""


def _has_article_ref(q: str) -> bool:
    import re
    if re.search(r"第[一二三四五六七八九十百千万零0-9]{1,12}条", q):
        return True
    if re.search(r"第[一二三四五六七八九十百千万零0-9]{1,12}(条|款|项|目)", q):
        return True
    if re.search(r"\barticle\s+\d{1,4}\b", q, flags=re.IGNORECASE):
        return True
    return False


def _extract_json(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return "{}"
