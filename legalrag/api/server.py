from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Any
from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline
from legalrag.utils.logger import get_logger
from legalrag.ingest.ingestor import PDFIngestor
from pathlib import Path
from legalrag.retrieval.incremental_indexer import IncrementalIndexer
from legalrag.retrieval.bm25_retriever import BM25Retriever
from filelock import FileLock
from fastapi import BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os
from starlette.responses import FileResponse
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

logger = get_logger(__name__)

app = FastAPI(title="Legal-RAG API", version="0.2.0")
CFG: AppConfig | None = None
pipeline: RagPipeline | None = None

def load_cfg() -> AppConfig:
    cfg = AppConfig.load()

    # 1) 读取 env 
    qwen_model = os.getenv(cfg.llm.qwen_model_env, "").strip()
    if qwen_model:
        cfg.llm.qwen_model = qwen_model

    openai_model = os.getenv(cfg.llm.openai_model_env, "").strip()
    if openai_model:
        cfg.llm.openai_model = openai_model

    # 2) provider 决策：有 key 就 openai，否则 qwen-local
    key_env = cfg.llm.api_key_env
    openai_key = os.getenv(key_env, "").strip()
    if openai_key:
        cfg.llm.provider = "openai"
        chosen = cfg.llm.openai_model
    else:
        cfg.llm.provider = "qwen-local"
        chosen = cfg.llm.qwen_model

    # 3) 将“最终生效模型”写回一个统一字段 
    cfg.llm.model = chosen

    logger.info(f"[LLM] provider={cfg.llm.provider}, model={cfg.llm.model}")
    return cfg


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


def _get(obj: Any, key: str, default=None):
    """Support dict + pydantic model + dataclass-like objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    # pydantic v1: .dict(); v2: .model_dump()
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump().get(key, default)
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict().get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)



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
    global CFG, pipeline
    try:
        logger.info("[API] 初始化 RAG Pipeline...")
        CFG = load_cfg()
        pipeline = RagPipeline(CFG)
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
# RAG Query Endpoint
# -----------------------
@app.post("/rag/query")
async def rag_query(body: dict):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="请求中缺少 'question' 字段")

    top_k = int(body.get("top_k", 10))
    answer_style = body.get("answer_style", "专业")

    try:
        ans = pipeline.answer(question, top_k=top_k)  # 如果你后面接 answer_style，可加进去

        # RagAnswer.answer 通常是 str
        answer_text = _get(ans, "answer", "")
        # 有些实现可能叫 text
        if not answer_text:
            answer_text = _get(ans, "text", "")

        hits = _get(ans, "hits", []) or []

        hit_rows = []
        for h in hits:
            # h 可能是 RetrievalHit 对象或 dict
            chunk = _get(h, "chunk", None)
            hit_rows.append(
                {
                    "rank": _get(h, "rank", ""),
                    "score": _get(h, "score", 0.0),
                    "law_name": _get(chunk, "law_name", "") if chunk else "",
                    "chapter": _get(chunk, "chapter", "") if chunk else "",
                    "section": _get(chunk, "section", "") if chunk else "",
                    "article_no": _get(chunk, "article_no", "") if chunk else "",
                    "text": _get(chunk, "text", "") if chunk else "",
                }
            )

        return {
            "question": question,
            "answer": answer_text,
            "hits": hit_rows,
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
        added = IncrementalIndexer(CFG).add_jsonl(jsonl_path)
        INGEST_STATUS.setdefault(doc_id, {})
        INGEST_STATUS[doc_id].update({"faiss": "done", "added": int(added)})
        logger.info(f"[FAISS] indexed doc_id={doc_id} added={added} jsonl={jsonl_path}")
    except Exception as e:
        INGEST_STATUS.setdefault(doc_id, {})
        INGEST_STATUS[doc_id].update({"faiss": "failed", "error": str(e)})
        logger.exception(f"[FAISS] indexing failed doc_id={doc_id} jsonl={jsonl_path}: {e}")

def bm25_rebuild_job(doc_id: str):
    try:
        lock = FileLock(str(Path(CFG.retrieval.bm25_index_file).with_suffix(".lock")))
        with lock:
            BM25Retriever(CFG).build()
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

    tmp_path = Path(CFG.paths.upload_dir) / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as f:
        f.write(contents)

    try:
        ingestor = PDFIngestor(CFG)
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

