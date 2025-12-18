from __future__ import annotations

import asyncio
import os, subprocess
from pathlib import Path
from typing import Any, Optional
import threading, time, requests
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock
import re
from legalrag.config import AppConfig
from legalrag.ingest.ingestor import PDFIngestor
from legalrag.llm.client import LLMClient
from legalrag.pipeline.rag_pipeline import RagPipeline
from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.retrieval.incremental_indexer import IncrementalIndexer
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Legal-RAG API", version="0.2.0")

CFG: Optional[AppConfig] = None
pipeline: Optional[RagPipeline] = None

# Pipeline init state (do NOT use these to gate /health or /)
PIPELINE_READY = False
PIPELINE_ERROR: Optional[str] = None


def detect_gpu() -> bool:
    """Fast, safe GPU detection for status (never raises)."""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_cfg() -> AppConfig:
    """
    Load config and decide provider/model:
    - If GPU is available: qwen-local
    - Else if server has OPENAI_API_KEY: openai
    - Else: disabled (wait for user key from UI)
    """
    cfg = AppConfig.load()

    # Allow env overrides for model names
    qwen_model = os.getenv(cfg.llm.qwen_model_env, "").strip()
    if qwen_model:
        cfg.llm.qwen_model = qwen_model

    openai_model = os.getenv(cfg.llm.openai_model_env, "").strip()
    if openai_model:
        cfg.llm.openai_model = openai_model

    openai_key = os.getenv(cfg.llm.api_key_env, "").strip()
    has_gpu = detect_gpu()

    if has_gpu:
        cfg.llm.provider = "qwen-local"
        chosen = cfg.llm.qwen_model
    else:
        if openai_key:
            cfg.llm.provider = "openai"
            chosen = cfg.llm.openai_model
        else:
            cfg.llm.provider = "disabled"
            chosen = ""

    # Optional: some configs may not have cfg.llm.model; ignore if absent.
    try:
        cfg.llm.model = chosen  # type: ignore[attr-defined]
    except Exception:
        pass

    logger.info(f"[LLM] provider={cfg.llm.provider}, model={chosen}, has_gpu={has_gpu}")
    return cfg


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # /app
UI_PATH = BASE_DIR / "ui"
logger.info(f"[boot] UI_PATH={UI_PATH} exists={UI_PATH.exists()}")

@app.get("/", include_in_schema=False)
def root():
    index = UI_PATH / "index.html"
    # 让默认页面就是 UI
    if index.exists():
        return FileResponse(str(index))
    # fallback 
    return HTMLResponse("<html><body>ok</body></html>", status_code=200)


@app.get("/health", include_in_schema=False)
def health():
    # Liveness: must always be fast and 200.
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
def ready():
    # Readiness: informational (can be false while warming up).
    return {
        "ready": bool(PIPELINE_READY),
        "error": PIPELINE_ERROR,
        "has_gpu": detect_gpu(),
        "provider": getattr(getattr(CFG, "llm", None), "provider", None),
    }

if UI_PATH.exists():
    # Serve static UI under /ui
    app.mount("/ui", StaticFiles(directory=str(UI_PATH), html=True), name="ui")

    # Ensure /ui/ returns the index.html explicitly (some platforms/probes prefer explicit route)
    @app.get("/ui/", include_in_schema=False)
    def ui_index():
        index = UI_PATH / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return HTMLResponse("<html><body>UI not found</body></html>", status_code=404)

    logger.info("[boot] UI configured at /ui/ (static).")


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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(exc)})


def build_pipeline_sync():
    """Build pipeline in a thread."""
    global CFG, pipeline, PIPELINE_READY, PIPELINE_ERROR
    PIPELINE_READY = False
    PIPELINE_ERROR = None
    try:
        logger.info("[API] 初始化 RAG Pipeline...")
        CFG = load_cfg()
        pipeline = RagPipeline(CFG)
        PIPELINE_READY = True
        logger.info("[API] RAG Pipeline 初始化完成")
    except Exception as e:
        PIPELINE_ERROR = repr(e)
        pipeline = None
        logger.exception("[API] RAG Pipeline 初始化失败")


@app.on_event("startup")
async def startup_event():
    # Do not block startup; allow / and /health to come up immediately.
    def _selfcheck():
        time.sleep(2)
        port = int(os.getenv("PORT") or "7860")
        for p in ["/", "/health", "/ui/", "/ui/index.html"]:
            try:
                r = requests.get(f"http://127.0.0.1:{port}{p}", timeout=2)
                logger.info(f"[selfcheck] GET {p} -> {r.status_code}")
            except Exception as e:
                logger.error(f"[selfcheck] GET {p} failed: {e}")

    threading.Thread(target=_selfcheck, daemon=True).start()
    logger.info("[net] " + subprocess.getoutput("ss -ltnp | head -n 20"))
    asyncio.create_task(asyncio.to_thread(build_pipeline_sync))


@app.post("/rag/query")
async def rag_query(body: dict, request: Request):
    if not PIPELINE_READY or pipeline is None or CFG is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is warming up")

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    user_openai_key = request.headers.get("X-OpenAI-Api-Key", "").strip()
    logger.info(f"[req] X-OpenAI-Api-Key present={bool(user_openai_key)} len={len(user_openai_key)}")

    llm_override = None

    if getattr(CFG.llm, "provider", "") == "disabled":
        if not user_openai_key:
            raise HTTPException(status_code=401, detail="OpenAI API Key required. Please enter it in the UI.")
        llm_override = LLMClient.from_config_with_key(CFG, openai_key=user_openai_key)

    elif getattr(CFG.llm, "provider", "") == "openai":
        if not os.getenv(CFG.llm.api_key_env, "").strip():
            if not user_openai_key:
                raise HTTPException(status_code=401, detail="OpenAI API Key required. Please enter it in the UI.")
            llm_override = LLMClient.from_config_with_key(CFG, openai_key=user_openai_key)

    # log override safely
    logger.info(
        f"[req] llm_override_provider={getattr(llm_override, 'provider', None)} "
        f"model={getattr(llm_override, 'model', None)}"
    )

    try:
        ans = pipeline.answer(question, llm_override=llm_override)

        answer_text = _get(ans, "answer", "") or _get(ans, "text", "")
        hits = _get(ans, "hits", []) or []

        hit_rows = []
        for h in hits:
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

        return {"question": question, "answer": answer_text, "hits": hit_rows}

    except HTTPException:
        raise
    except Exception as e:
        # Classify common OpenAI/client failures so the frontend can show friendly messages
        msg = str(e) if e is not None else ""
        msg_l = msg.lower()

        # 1) Insufficient quota / rate limit
        if ("insufficient_quota" in msg_l) or ("you exceeded your current quota" in msg_l):
            logger.warning(f"RAG query failed (insufficient_quota): {msg}")
            raise HTTPException(
                status_code=429,
                detail={
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                    "message": msg,
                },
            )

        # 2) Missing model parameter
        if ("must provide a model parameter" in msg_l) or ("provide a model parameter" in msg_l):
            logger.warning(f"RAG query failed (missing_model): {msg}")
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "missing_model",
                    "code": "missing_model",
                    "message": msg,
                },
            )

        # 3) Missing API key 
        if ("api key required" in msg_l) or ("openai api key required" in msg_l):
            logger.warning(f"RAG query failed (missing_api_key): {msg}")
            raise HTTPException(
                status_code=401,
                detail={
                    "type": "missing_api_key",
                    "code": "missing_api_key",
                    "message": msg,
                },
            )

        # 4) Generic OpenAI 4xx/5xx surfaced in message like: "Error code: 429 - {...}"
        m = re.search(r"error code:\s*(\d{3})", msg_l)
        if m:
            code = int(m.group(1))
            logger.warning(f"RAG query failed (llm_error_{code}): {msg}")
            raise HTTPException(
                status_code=code if 400 <= code <= 599 else 500,
                detail={
                    "type": "llm_error",
                    "code": code,
                    "message": msg,
                },
            )

        logger.exception(f"RAG 查询失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "type": "server_error",
                "message": "RAG query failed",
                "raw": msg[:2000],  # avoid overly large payloads
            },
        )



# -----------------------
# PDF ingest
# -----------------------
INGEST_STATUS = {}  # doc_id -> {"faiss": "...", "added": int, "bm25": "...", "error": str|None}


def faiss_index_job(jsonl_path: str, doc_id: str):
    try:
        assert CFG is not None
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
        assert CFG is not None
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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    if CFG is None:
        raise HTTPException(status_code=503, detail="Server is warming up")

    contents = await file.read()
    max_mb = 10
    if len(contents) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {max_mb}MB limit")

    tmp_path = Path(CFG.paths.upload_dir) / file.filename
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(contents)

    try:
        ingestor = PDFIngestor(CFG)
        result = ingestor.ingest_pdf_to_jsonl(tmp_path, law_name=f"用户上传PDF-{file.filename}")

        if background_tasks is not None:
            INGEST_STATUS[result.doc_id] = {"faiss": "scheduled", "added": 0, "bm25": "scheduled", "error": None}
            background_tasks.add_task(faiss_index_job, result.jsonl_path, result.doc_id)
            background_tasks.add_task(bm25_rebuild_job, result.doc_id)

        return {
            "filename": file.filename,
            "doc_id": result.doc_id,
            "num_chunks": result.num_chunks,
            "text_preview": result.preview,
            "jsonl_path": result.jsonl_path,
            "index_status": "scheduled",
        }
    except Exception as e:
        logger.exception(f"PDF ingest failed: {e}")
        raise HTTPException(status_code=500, detail="PDF ingest failed")


@app.get("/ingest/status/{doc_id}")
def ingest_status(doc_id: str):
    return INGEST_STATUS.get(doc_id, {"error": "not_found"})
