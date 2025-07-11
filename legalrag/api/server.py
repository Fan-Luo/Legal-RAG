from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline
from legalrag.utils.logger import get_logger
from legalrag.ingest.ingestor import PDFIngestor
from pathlib import Path
from legalrag.retrieval.incremental_indexer import IncrementalIndexer
from legalrag.retrieval.bm25_retriever import BM25Retriever
from filelock import FileLock
from fastapi import BackgroundTasks

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


@app.get("/")
def root():
    return JSONResponse(
        {
            "name": "Legal-RAG",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
        }
    )

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

    top_k = body.get("top_k", 10)
    answer_style = body.get("answer_style", "专业")

    try:
        ans = await pipeline.answer_async(
            question,
            top_k=int(top_k),
            answer_style=answer_style,
        )
        return {
            "question": question,
            "answer": ans["text"],
            "mode": ans["mode"],
            "provider": ans["provider"],
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

INGEST_STATUS = {}  # doc_id -> {"faiss": "...", "added": int, "bm25": "...", "error": str|None}
def faiss_index_job(jsonl_path: str, doc_id: str):
    try:
        added = IncrementalIndexer(cfg).add_jsonl(jsonl_path)
        INGEST_STATUS.setdefault(doc_id, {})
        INGEST_STATUS[doc_id].update({"faiss": "done", "added": int(added)})
        logger.info(f"[FAISS] indexed doc_id={doc_id} added={added} jsonl={jsonl_path}")
    except Exception as e:
        INGEST_STATUS.setdefault(doc_id, {})
        INGEST_STATUS[doc_id].update({"faiss": "failed", "error": str(e)})
        logger.exception(f"[FAISS] indexing failed doc_id={doc_id} jsonl={jsonl_path}: {e}")

def bm25_rebuild_job(doc_id: str):
    try:
        lock = FileLock(str(Path(cfg.retrieval.bm25_index_file).with_suffix(".lock")))
        with lock:
            BM25Retriever(cfg).build()
        INGEST_STATUS.setdefault(doc_id, {})
        INGEST_STATUS[doc_id].update({"bm25": "done"})
        logger.info(f"[BM25] rebuild done doc_id={doc_id}")
    except Exception as e:
        INGEST_STATUS.setdefault(doc_id, {})
        INGEST_STATUS[doc_id].update({"bm25": "failed", "error": str(e)})
        logger.exception(f"[BM25] rebuild failed doc_id={doc_id}: {e}")


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    contents = await file.read()
    MAX_FILE_SIZE_MB = 10
    file_size = len(contents)
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"文件大小{file_size/ 1024 / 1024}MB 超过限制（最大 {MAX_FILE_SIZE_MB}MB）")

    tmp_path = Path(cfg.paths.upload_dir) / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as f:
        f.write(contents)

    try:
        ingestor = PDFIngestor(cfg)
        result = ingestor.ingest_pdf_to_jsonl(tmp_path, law_name=f"用户上传PDF-{file.filename}")

        # 把索引放到后台，不阻塞返回
        if background_tasks is not None:
            INGEST_STATUS[result.doc_id] = {"faiss": "scheduled", "added": 0, "bm25": "scheduled", "error": None}
            background_tasks.add_task(faiss_index_job, result.jsonl_path, result.doc_id)
            background_tasks.add_task(bm25_rebuild_job, result.doc_id)

        # 立即返回 preview + 状态
        return {
            "filename": file.filename,
            "doc_id": result.doc_id,
            "num_chunks": result.num_chunks,
            "text_preview": result.preview,     
            "jsonl_path": result.jsonl_path,
            "index_status": "scheduled",         
        }
    except Exception as e:
        logger.exception(f"PDF 解析失败: {e}")
        raise HTTPException(status_code=500, detail="PDF 解析失败")

@app.get("/ingest/status/{doc_id}")
def ingest_status(doc_id: str):
    return INGEST_STATUS.get(doc_id, {"error": "not_found"})

