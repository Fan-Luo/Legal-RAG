from __future__ import annotations

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline
from legalrag.pdf.parser import extract_text_from_pdf
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Legal-RAG API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

cfg = AppConfig.load()
pipeline: RagPipeline | None = None


@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info("[API] 初始化 RAG Pipeline...")
    pipeline = RagPipeline(cfg)
    logger.info("[API] RAG Pipeline 初始化完成")


@app.get("/health")
def health():
    return {"status": "ok"}