from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.llm.gateway import LLMGateway
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.routing.router import QueryRouter, RoutingDecision
from legalrag.schemas import RetrievalHit
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Legal-RAG Retrieval Service", version="0.1.0")

CFG: Optional[AppConfig] = None
RETRIEVER: Optional[HybridRetriever] = None
ROUTER: Optional[QueryRouter] = None


def _serialize_hits(hits: List[RetrievalHit]) -> List[Dict[str, Any]]:
    rows = []
    for h in hits:
        rows.append(h.model_dump())
    return rows


@app.on_event("startup")
def _startup() -> None:
    global CFG, RETRIEVER, ROUTER
    CFG = AppConfig.load()
    RETRIEVER = HybridRetriever(CFG)
    llm = LLMGateway(
        LLMClient.from_config(CFG),
        request_timeout=CFG.llm.request_timeout,
        max_retries=CFG.llm.max_retries,
        retry_backoff=CFG.llm.retry_backoff,
    )
    ROUTER = QueryRouter(llm_client=llm, llm_based=CFG.routing.llm_based, cfg=CFG)
    logger.info("retrieval service ready")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(body: Dict[str, Any]):
    if RETRIEVER is None or ROUTER is None or CFG is None:
        raise HTTPException(status_code=503, detail="retriever not ready")

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    top_k = body.get("top_k")
    try:
        top_k = int(top_k) if top_k is not None else CFG.retrieval.top_k
    except Exception:
        top_k = CFG.retrieval.top_k

    decision: RoutingDecision = ROUTER.route(question)
    eff_top_k = int(top_k * getattr(decision, "top_k_factor", 1.0))
    eff_top_k = max(3, min(eff_top_k, 30))

    hits = RETRIEVER.search(question, top_k=eff_top_k, decision=decision)

    return {
        "question": question,
        "top_k": eff_top_k,
        "decision": decision.model_dump(),
        "hits": _serialize_hits(hits),
    }
