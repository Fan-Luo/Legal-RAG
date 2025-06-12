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


@app.post("/rag/query")
async def rag_query(body: dict):
    assert pipeline is not None
    question = body.get("question", "")
    ans = pipeline.answer(question)
    return {
        "question": ans.question,
        "answer": ans.answer,
        "hits": [
            {
                "rank": h.rank,
                "score": h.score,
                "law_name": h.chunk.law_name,
                "chapter": h.chunk.chapter,
                "section": h.chunk.section,
                "article_no": h.chunk.article_no,
                "text": h.chunk.text,
            }
            for h in ans.hits
        ],
    }


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    tmp_path = cfg.paths.upload_dir + "/" + file.filename
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    text = extract_text_from_pdf(tmp_path, cfg)
    return {
        "filename": file.filename,
        "text_preview": text[:800],
        "length": len(text),
    }
