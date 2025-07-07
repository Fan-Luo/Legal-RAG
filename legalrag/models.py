from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class LawChunk(BaseModel):
    id: str
    law_name: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    article_no: str
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
    explain: Optional[Dict[str, Any]] = None

    @field_validator("chunk", mode="before")
    @classmethod
    def _coerce_chunk(cls, v):
        """
        Notebook/热重载场景下，可能出现“旧 LawChunk 实例”：
        - v 是 dict：直接 validate
        - v 是任意 pydantic BaseModel：dump -> validate
        - v 是 LawChunk：直接返回
        """
        if isinstance(v, LawChunk):
            return v
        if isinstance(v, BaseModel):
            return LawChunk.model_validate(v.model_dump())
        if isinstance(v, dict):
            return LawChunk.model_validate(v)
        return v


class QueryType(str, Enum):
    DEFINITION = "definition"
    LIABILITY = "liability"
    VALIDITY = "validity"
    PROCEDURE = "procedure"
    TERMINATION = "termination"
    DAMAGE = "damage"
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


class LawNode(BaseModel):
    article_id: str
    article_no: str
    title: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    neighbors: List[str] = []
    meta: Dict[str, Any] = {}

    # Pydantic v2 uses model_config
    class Config:
        arbitrary_types_allowed = True


# Pydantic v2: resolve forward refs / Literal reliably
try:
    LawChunk.model_rebuild()
    RetrievalHit.model_rebuild()
    RoutingDecision.model_rebuild()
    RagAnswer.model_rebuild()
    LawNode.model_rebuild()
except Exception:
    pass
