from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException

from legalrag.config import AppConfig
from legalrag.index.registry import IndexRegistry

app = FastAPI(title="Legal-RAG Index Service", version="0.1.0")

CFG: Optional[AppConfig] = None
REGISTRY: Optional[IndexRegistry] = None


@app.on_event("startup")
def _startup() -> None:
    global CFG, REGISTRY
    CFG = AppConfig.load()
    REGISTRY = IndexRegistry(Path(CFG.paths.index_dir))


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/index/active")
def active() -> Dict[str, str]:
    if REGISTRY is None:
        raise HTTPException(status_code=503, detail="registry not ready")
    return {"active": REGISTRY.active_version() or "default"}


@app.get("/index/list")
def list_versions():
    if REGISTRY is None:
        raise HTTPException(status_code=503, detail="registry not ready")
    return {"versions": REGISTRY.list_versions()}


@app.post("/index/activate/{version}")
def activate(version: str):
    if REGISTRY is None:
        raise HTTPException(status_code=503, detail="registry not ready")
    try:
        REGISTRY.activate(version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"active": version}
