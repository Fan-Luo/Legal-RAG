from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


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
    chunk: LawChunk
    score: float
    rank: int


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

    class Config:
        arbitrary_types_allowed = True
