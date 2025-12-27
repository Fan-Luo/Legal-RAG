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

class QueryType(str, Enum):
    """
    High-level legal question categories for contract law RAG.
    """

    DEFINITION = "definition"  # 概念 / 定义 / 构成要件
    VALIDITY = "validity"      # 合同成立、生效、无效、可撤销（欺诈、胁迫、显失公平等）
    PERFORMANCE = "performance"    # 履行、抗辩、价款、风险承担、瑕疵履行等
    BREACH_REMEDY = "breach_remedy"    # 违约责任、违约金、损害赔偿、定金、继续履行等救济
    TERMINATION = "termination"    # 合同解除、终止、解除后果
    PROCEDURE = "procedure"    # 诉讼时效、期间起算、中断中止、举证责任、程序性问题
    OTHER = "other"   

class RoutingMode(str, Enum):
    DIRECT_LLM = "direct_llm"
    RAG = "rag"
    GRAPH_AUGMENTED = "graph_augmented"

class RoutingDecision(BaseModel):
    query_type: QueryType
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
