from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# -----------------------
# Global Exception Handler
# -----------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

# -----------------------
# Startup Event
# -----------------------
@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        logger.info("[API] 初始化 RAG Pipeline...")
        pipeline = RagPipeline(cfg)
        logger.info("[API] RAG Pipeline 初始化完成")
    except Exception as e:
        logger.exception("[API] RAG Pipeline 初始化失败")
        # Optionally raise to prevent API from starting
        raise RuntimeError("RAG Pipeline 初始化失败") from e

# -----------------------
# Health Check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
# RAG Query Endpoint
# -----------------------
@app.post("/rag/query")
async def rag_query(body: dict):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="请求中缺少 'question' 字段")
    
    try:
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
    except Exception as e:
        logger.exception(f"RAG 查询失败: {e}")
        raise HTTPException(status_code=500, detail="RAG 查询失败")

# -----------------------
# PDF Ingest Endpoint
# -----------------------
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件上传")

    tmp_path = f"{cfg.paths.upload_dir}/{file.filename}"
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        text = extract_text_from_pdf(tmp_path, cfg)
        return {
            "filename": file.filename,
            "text_preview": text[:800],
            "length": len(text),
        }
    except Exception as e:
        logger.exception(f"PDF 解析失败: {e}")
        raise HTTPException(status_code=500, detail="PDF 解析失败")
