"""
LegalIssueExtractor: lightweight civil-law issue classifier (CN + EN).

- Heuristic-first; no ML dependencies required.
- Optional LLM refinement via cfg.routing.issue_llm_refine (default False).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LegalIssueType(str, Enum):
    # Contract
    CONTRACT_FORMATION = "contract_formation"
    CONTRACT_INTERPRETATION = "contract_interpretation"
    CONTRACT_BREACH = "contract_breach"
    CONTRACT_REMEDIES = "contract_remedies"
    SURETY_GUARANTEE = "surety_guarantee"

    # Tort
    TORT_LIABILITY = "tort_liability"
    PERSONAL_INJURY = "personal_injury"
    PRODUCT_LIABILITY = "product_liability"
    PRIVACY_DEFAMATION = "privacy_defamation"

    # Property
    PROPERTY_OWNERSHIP = "property_ownership"
    POSSESSION = "possession"
    REAL_ESTATE = "real_estate"
    MORTGAGE = "mortgage"

    # Family
    MARRIAGE_FAMILY = "marriage_family"
    DIVORCE = "divorce"
    CUSTODY = "custody"
    MAINTENANCE = "maintenance"

    # Inheritance
    INHERITANCE = "inheritance"
    WILL_SUCCESSION = "will_succession"

    # Personality rights
    PERSONALITY_RIGHTS = "personality_rights"

    # Others
    AGENCY = "agency"
    UNJUST_ENRICHMENT = "unjust_enrichment"
    OTHER = "other"


class LegalIssueResult(BaseModel):
    legal_issue_type: LegalIssueType = LegalIssueType.OTHER
    tags: List[str] = Field(default_factory=list)
    explain: str = ""
    signals: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class LegalIssueExtractor:
    llm: Optional[Any] = None
    cfg: Optional[Any] = None

    def extract(self, question: str) -> LegalIssueResult:
        q = (question or "").strip()
        s = q.lower()

        tags: List[str] = []
        signals: Dict[str, Any] = {}

        has_article_ref = _has_article_ref(q)
        signals["has_article_ref"] = has_article_ref

        # CN keywords
        contract_kw = ["合同", "违约", "解除", "履行", "定金", "订金", "违约金", "损害赔偿", "交付", "瑕疵", "质量", "付款", "退货", "保证", "担保"]
        tort_kw = ["侵权", "过错", "过失", "故意", "人身损害", "精神损害", "名誉", "诽谤", "隐私", "肖像", "产品责任", "缺陷产品"]
        property_kw = ["物权", "所有权", "占有", "不动产", "房屋", "土地", "登记", "共有", "相邻关系", "抵押"]
        family_kw = ["婚姻", "离婚", "夫妻", "共同财产", "抚养", "抚养费", "探望", "监护", "家暴"]
        inherit_kw = ["继承", "遗嘱", "遗产", "法定继承", "遗赠", "代位继承"]
        unjust_kw = ["不当得利", "返还", "无因管理"]
        agency_kw = ["代理", "委托", "授权", "表见代理"]

        # EN keywords
        contract_en = ["contract", "breach", "terminate", "rescission", "liquidated", "damages", "warranty", "defective", "delivery", "refund", "deposit", "guarantee", "surety"]
        tort_en = ["tort", "negligence", "fault", "defamation", "privacy", "injury", "liability", "product liability"]
        property_en = ["property", "ownership", "possession", "real estate", "mortgage", "title", "registration"]
        family_en = ["divorce", "marriage", "custody", "alimony", "maintenance", "domestic violence"]
        inherit_en = ["inheritance", "will", "estate", "succession"]
        unjust_en = ["unjust enrichment"]
        agency_en = ["agency", "power of attorney", "authorization", "apparent authority"]

        scores = {
            "contract": _kw_score(s, contract_kw) + _kw_score(s, contract_en),
            "tort": _kw_score(s, tort_kw) + _kw_score(s, tort_en),
            "property": _kw_score(s, property_kw) + _kw_score(s, property_en),
            "family": _kw_score(s, family_kw) + _kw_score(s, family_en),
            "inherit": _kw_score(s, inherit_kw) + _kw_score(s, inherit_en),
            "unjust": _kw_score(s, unjust_kw) + _kw_score(s, unjust_en),
            "agency": _kw_score(s, agency_kw) + _kw_score(s, agency_en),
        }
        top_domain, top_score = max(scores.items(), key=lambda x: float(x[1]))
        signals["domain_scores"] = dict(scores)

        t = LegalIssueType.OTHER
        if top_score <= 0:
            t = LegalIssueType.OTHER
        elif top_domain == "contract":
            if any(k in s for k in ["订立", "成立", "要约", "承诺", "格式条款", "无效", "撤销", "formation"]):
                t = LegalIssueType.CONTRACT_FORMATION
            elif any(k in s for k in ["解释", "如何理解", "interpret"]):
                t = LegalIssueType.CONTRACT_INTERPRETATION
            elif any(k in s for k in ["解除", "赔偿", "违约金", "退货", "refund", "damages", "terminate"]):
                t = LegalIssueType.CONTRACT_REMEDIES
            elif any(k in s for k in ["保证", "担保", "surety", "guarantee"]):
                t = LegalIssueType.SURETY_GUARANTEE
            else:
                t = LegalIssueType.CONTRACT_BREACH
        elif top_domain == "tort":
            if any(k in s for k in ["产品", "缺陷", "product", "defect"]):
                t = LegalIssueType.PRODUCT_LIABILITY
            elif any(k in s for k in ["人身", "injury", "伤", "医疗", "交通事故"]):
                t = LegalIssueType.PERSONAL_INJURY
            elif any(k in s for k in ["名誉", "诽谤", "隐私", "肖像", "defamation", "privacy"]):
                t = LegalIssueType.PRIVACY_DEFAMATION
            else:
                t = LegalIssueType.TORT_LIABILITY
        elif top_domain == "property":
            if any(k in s for k in ["房", "房屋", "不动产", "土地", "real estate"]):
                t = LegalIssueType.REAL_ESTATE
            elif "抵押" in s or "mortgage" in s:
                t = LegalIssueType.MORTGAGE
            elif any(k in s for k in ["占有", "possession"]):
                t = LegalIssueType.POSSESSION
            else:
                t = LegalIssueType.PROPERTY_OWNERSHIP
        elif top_domain == "family":
            if "离婚" in s or "divorce" in s:
                t = LegalIssueType.DIVORCE
            elif "抚养" in s or "custody" in s:
                t = LegalIssueType.CUSTODY
            elif "抚养费" in s or "alimony" in s or "maintenance" in s:
                t = LegalIssueType.MAINTENANCE
            else:
                t = LegalIssueType.MARRIAGE_FAMILY
        elif top_domain == "inherit":
            if "遗嘱" in s or "will" in s:
                t = LegalIssueType.WILL_SUCCESSION
            else:
                t = LegalIssueType.INHERITANCE
        elif top_domain == "unjust":
            t = LegalIssueType.UNJUST_ENRICHMENT
        elif top_domain == "agency":
            t = LegalIssueType.AGENCY

        tags.append(top_domain)
        if has_article_ref:
            tags.append("article_ref")

        explain = f"heuristic_domain={top_domain} score={top_score}; legal_issue_type={t}"
        out = LegalIssueResult(legal_issue_type=t, tags=tags, explain=explain, signals=signals)

        # Optional LLM refine (off by default)
        if self.llm is not None and bool(getattr(getattr(self.cfg, "routing", None), "issue_llm_refine", False)):
            try:
                out = self._llm_refine(question, out)
            except Exception:
                return out
        return out

    def _llm_refine(self, question: str, base: LegalIssueResult) -> LegalIssueResult:
        llm = self.llm
        if llm is None:
            return base

        sys = (
            "Classify the user question into a civil-law issue type. "
            "Return ONLY JSON with keys: legal_issue_type, tags. "
            f"legal_issue_type must be one of: {[e.value for e in LegalIssueType]}."
        )
        user = {"question": question, "heuristic": base.model_dump()}
        text = str(llm.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": str(user)}]))

        import json
        obj = json.loads(_extract_json(text))
        t = str(obj.get("legal_issue_type", "")).strip()
        tags = obj.get("tags", []) or []
        if t in [e.value for e in LegalIssueType]:
            base.legal_issue_type = LegalIssueType(t)
        if isinstance(tags, list):
            base.tags = [str(x) for x in tags if str(x)]
        base.explain = (base.explain + "; llm_refine_ok").strip("; ")
        return base


def _kw_score(text: str, kws: List[str]) -> int:
    return sum(1 for k in kws if k and k.lower() in text)


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
