from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import field
from pydantic import BaseModel, ConfigDict, field_validator 


class LawChunk(BaseModel):
    id: str
    law_name: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    article_no: str
    article_id: str  # article_no 的数字编号
    text: str
    lang: Optional[str] = "zh"
    source: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

class RetrievalHit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    chunk: LawChunk
    score: float
    rank: Optional[int] = None
    source: Literal["retriever", "graph", "rerank"] = "retriever"
    semantic_score: Optional[float] = None
    graph_depth: Optional[int] = None
    relations: Optional[List[str]] = None
    seed_article_id: Optional[str] = None
    score_breakdown: Optional[Dict[str, Any]] = None

class TaskType(str, Enum):
    """
    Task/output structure axis for prompting.
    """

    JUDGE_STYLE = "judge_style"
    STATUTE_EXEGESIS = "statute_exegesis"
    RISK_ALERT = "risk_alert"
    ELEMENTS_CHECKLIST = "elements_checklist"
    COMPARATIVE_RULES = "comparative_rules"
    PROCEDURE_EVIDENCE_LIST = "procedure_evidence_list"
    OTHER = "other"


class IssueType(str, Enum):
    """
    Legal issue axis for semantic classification.
    """

    GENERAL_CIVIL = "general_civil"
    CIVIL_CAPACITY = "civil_capacity"
    CIVIL_ACT_VALIDITY = "civil_act_validity"
    AGENCY = "agency"
    CIVIL_LIABILITY = "civil_liability"
    LIMITATION_PERIOD = "limitation_period"

    PROPERTY = "property"
    OWNERSHIP = "ownership"
    POSSESSION = "possession"
    REGISTRATION = "registration"
    NEIGHBOR_RELATION = "neighbor_relation"
    PROPERTY_USE_RIGHT = "property_use_right"
    MORTGAGE = "mortgage"
    PLEDGE = "pledge"
    LIEN = "lien"

    CONTRACT = "contract"
    CONTRACT_FORMATION = "contract_formation"
    CONTRACT_VALIDITY = "contract_validity"
    CONTRACT_INTERPRETATION = "contract_interpretation"
    CONTRACT_PERFORMANCE = "contract_performance"
    PERFORMANCE_DEFENSE = "performance_defense"
    DEFECTIVE_PERFORMANCE = "defective_performance"
    CONTRACT_TERMINATION = "contract_termination"
    BREACH_REMEDY = "breach_remedy"
    PENALTY_LIQUIDATED = "penalty_liquidated"
    DEPOSIT = "deposit"
    GUARANTEE = "guarantee"
    CONTRACT_TRANSFER = "contract_transfer"

    QUASI_CONTRACT = "quasi_contract"
    NEGOTIORUM_GESTIO = "negotiorum_gestio"
    UNJUST_ENRICHMENT = "unjust_enrichment"

    PERSONALITY = "personality"
    NAME_RIGHT = "name_right"
    PORTRAIT_RIGHT = "portrait_right"
    REPUTATION_RIGHT = "reputation_right"
    PRIVACY_INFO = "privacy_info"
    PERSONALITY_INFRINGEMENT = "personality_infringement"

    MARRIAGE_FAMILY = "marriage_family"
    MARRIAGE = "marriage"
    DIVORCE = "divorce"
    FAMILY_PROPERTY = "family_property"
    CUSTODY_SUPPORT = "custody_support"

    INHERITANCE = "inheritance"
    INHERITANCE_WILL = "inheritance_will"
    INHERITANCE_STATUTORY = "inheritance_statutory"
    INHERITANCE_SHARE = "inheritance_share"

    TORT = "tort"
    TORT_LIABILITY = "tort_liability"
    PERSONAL_INJURY = "personal_injury"
    PRODUCT_LIABILITY = "product_liability"
    MEDICAL_TORT = "medical_tort"
    OTHER = "other"

class RoutingMode(str, Enum):
    RAG = "RAG"
    GRAPH_AUGMENTED = "GRAPH_AUGMENTED"

class RoutingDecision(BaseModel):
    task_type: TaskType
    issue_type: IssueType
    mode: RoutingMode
    top_k_factor: float = 1.0

class RagAnswer(BaseModel):
    question: str
    answer: str
    hits: List[RetrievalHit]


class Neighbor(BaseModel):
    """A directed edge from one article node to another."""
    article_id: str
    relation: str = "neighbor"
    conf: float = 1.0
    evidence: Optional[Dict[str, Any]] = None

class LawNode(BaseModel):
    """Lightweight in-memory node. (Do NOT store query-time fields in JSONL.)"""
    article_id: str
    article_no: str = ""
    law_name: Optional[str] = None
    title: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    neighbors: List[Neighbor] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- query-time fields ----
    graph_depth: Optional[int] = None
    graph_parent: Optional[str] = None
    relations: Optional[str] = None

# Pydantic v2: resolve forward refs / Literal reliably
try:
    LawChunk.model_rebuild()
    RetrievalHit.model_rebuild()
    RoutingDecision.model_rebuild()
    RagAnswer.model_rebuild()
    LawNode.model_rebuild()
except Exception:
    pass
