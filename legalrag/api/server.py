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
        raise RuntimeError("RAG Pipeline 初始化失败") from e

# -----------------------
# Health Check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
# RAG Query Endpoint (async)
# -----------------------
@app.post("/rag/query")
async def rag_query(body: dict):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")

    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="请求中缺少 'question' 字段")

    try:
        # 调用异步 LLM
        ans = await pipeline.answer_async(question)

        # ans 是结构化 dict: text, mode, provider, context_snippet, hits
        return {
            "question": question,
            "answer": ans["text"],
            "mode": ans["mode"],                 # normal / degraded
            "provider": ans["provider"],         # qwen-local / openai
            "context_snippet": ans.get("context_snippet", ""),
            "hits": [
                {
                    "rank": h.rank,
                    "score": h.score,
                    "law_name": h.chunk.law_name,
                    "chapter": h.chunk.chapter or "",
                    "section": h.chunk.section or "",
                    "article_no": h.chunk.article_no,
                    "text": h.chunk.text,
                }
                for h in ans.get("hits", [])
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
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    # 获取文件大小
    file_size = len(await file.read())  
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件大小超过限制（最大 {MAX_FILE_SIZE_MB}MB）")
    

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
