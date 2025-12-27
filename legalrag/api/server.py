from __future__ import annotations

import asyncio
import os, subprocess, uuid
from pathlib import Path
from typing import Any, Optional
import threading, time, requests
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock
import time
from legalrag.config import AppConfig
from legalrag.llm.client import LLMClient
from legalrag.pipeline.rag_pipeline import RagPipeline
from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.retrieval.builders.incremental_builder import IncrementalDenseBuilder
from legalrag.ingest.service import IngestService

from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Legal-RAG API", version="0.2.0")

CFG: Optional[AppConfig] = None
pipeline: Optional[RagPipeline] = None
ingest_service: IngestService | None = None

PIPELINE_READY = False
PIPELINE_ERROR: Optional[str] = None


def detect_gpu() -> bool:
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

    try:
        cfg.llm.model = chosen   
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
    app.mount("/ui", StaticFiles(directory=str(UI_PATH), html=True), name="ui")

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
    ingest_service = IngestService(CFG)
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


# -----------------------
# Two-stage RAG  
# -----------------------

# In-memory cache for stage-1 retrieval results.
# retrieval_id -> {"question": str, "decision": Any, "hits": List[RetrievalHit], "created_at": float}
RETRIEVE_CACHE: dict[str, dict[str, Any]] = {}
RETRIEVE_TTL_SEC = 15 * 60  # 15 minutes


def _purge_retrieve_cache(now: float | None = None) -> None:
    now = now or time.time()
    dead = [rid for rid, v in RETRIEVE_CACHE.items() if (now - float(v.get("created_at", 0))) > RETRIEVE_TTL_SEC]
    for rid in dead:
        RETRIEVE_CACHE.pop(rid, None)


def _serialize_hits(hits: list[Any]) -> list[dict[str, Any]]:
    hit_rows: list[dict[str, Any]] = []
    for h in hits or []:
        chunk = _get(h, "chunk", None)
        hit_rows.append(
            {
                "rank": _get(h, "rank", ""),
                "score": float(_get(h, "score", 0.0) or 0.0),
                "law_name": _get(chunk, "law_name", "") if chunk else "",
                "chapter": _get(chunk, "chapter", "") if chunk else "",
                "section": _get(chunk, "section", "") if chunk else "",
                "article_no": _get(chunk, "article_no", "") if chunk else "",
                "text": _get(chunk, "text", "") if chunk else "",
                "source": _get(h, "source", "") or _get(chunk, "source", "") if chunk else "",
            }
        )
    return hit_rows


def _llm_override_from_request(request: Request) -> Optional[LLMClient]:
    """
    Create a per-request LLM override when:
      - server provider=disabled, or
      - server provider=openai but server has no env key
    Uses header: X-OpenAI-Api-Key
    """
    assert CFG is not None
    user_openai_key = request.headers.get("X-OpenAI-Api-Key", "").strip()

    if getattr(CFG.llm, "provider", "") == "disabled":
        if not user_openai_key:
            raise HTTPException(status_code=401, detail="OpenAI API Key required. Please enter it in the UI.")
        return LLMClient.from_config_with_key(CFG, openai_key=user_openai_key)

    if getattr(CFG.llm, "provider", "") == "openai":
        if not os.getenv(CFG.llm.api_key_env, "").strip():
            if not user_openai_key:
                raise HTTPException(status_code=401, detail="OpenAI API Key required. Please enter it in the UI.")
            return LLMClient.from_config_with_key(CFG, openai_key=user_openai_key)

    return None


@app.post("/rag/retrieve")
async def rag_retrieve(body: dict, request: Request):
    """
    Stage 1 (fast): retrieval only.
    Returns:
      - retrieval_id: cache key for the second stage
      - hits: citations for UI
    """
    if not PIPELINE_READY or pipeline is None or CFG is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is warming up")

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    top_k = body.get("top_k", None)
    try:
        top_k = int(top_k) if top_k is not None else None
    except Exception:
        top_k = None

    try:
        decision, hits, eff_top_k = pipeline.retrieve(question, top_k=top_k)

        rid = uuid.uuid4().hex
        _purge_retrieve_cache()
        RETRIEVE_CACHE[rid] = {
            "question": question,
            "decision": decision,
            "hits": hits,
            "created_at": time.time(),
        }

        return {
            "question": question,
            "retrieval_id": rid,
            "top_k": eff_top_k,
            "hits": _serialize_hits(hits),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"RAG retrieve failed: {e}")
        raise HTTPException(status_code=500, detail="RAG retrieve failed")


@app.post("/rag/answer")
async def rag_answer(body: dict, request: Request):
    """
    Stage 2: generate answer using cached hits.
    Supports SSE streaming when:
      - header Accept includes 'text/event-stream', OR
      - body['stream'] == True
    """
    if not PIPELINE_READY or pipeline is None or CFG is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is warming up")

    accept = (request.headers.get("accept") or "").lower()
    stream = bool(body.get("stream")) or ("text/event-stream" in accept)

    rid = (body.get("retrieval_id") or "").strip()
    question = (body.get("question") or "").strip()

    decision = None
    hits = None

    if rid:
        _purge_retrieve_cache()
        cached = RETRIEVE_CACHE.get(rid)
        if not cached:
            raise HTTPException(status_code=404, detail="retrieval_id not found or expired")
        question = cached.get("question", question) or question
        decision = cached.get("decision")
        hits = cached.get("hits") or []

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    if hits is None:
        # Stateless fallback is intentionally unsupported for now (hits are complex objects).
        hits = body.get("hits") or []
        if hits and isinstance(hits[0], dict):
            raise HTTPException(
                status_code=400,
                detail="Please call /rag/retrieve first and pass retrieval_id to /rag/answer",
            )

    llm_override = _llm_override_from_request(request)

    # -----------------------------
    # 返回JSON
    # -----------------------------
    if not stream:
        try:
            logger.info(f"answer_from_hits")
            ans = pipeline.answer_from_hits(
                question, hits, decision=decision, llm_override=llm_override
            )
            answer_text = _get(ans, "answer", "") or _get(ans, "text", "")
            out_hits = _get(ans, "hits", []) or hits or []
            return {
                "question": question,
                "answer": answer_text,
                "hits": _serialize_hits(out_hits),
                "retrieval_id": rid or None,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"RAG answer failed: {e}")
            raise HTTPException(status_code=500, detail="RAG answer failed")

    # -----------------------------
    # 流式返回（SSE）
    # -----------------------------
    import json as _json

    def _sse(event: str, data_obj: Any) -> bytes:
        if isinstance(data_obj, str):
            data_str = data_obj
        else:
            data_str = _json.dumps(data_obj, ensure_ascii=False)
        return f"event: {event}\ndata: {data_str}\n\n".encode("utf-8")

    async def gen():
        logger.info(f"gen()")
        yield b":" + b" " * 2048 + b"\n\n"
        # 强制 SSE 首次 flush
        yield b": ping\n\n"

        t0 = time.time()
        # yield _sse("meta", {"t0": t0, "retrieval_id": rid or None})

        # 再次 ping，确保中间层不缓冲
        yield b": ping\n\n"

        try:
            # 发送 meta 信息
            yield _sse("meta", {"question": question, "retrieval_id": rid or None})

            # 优先使用 pipeline 的异步流式生成函数
            stream_fn = getattr(pipeline, "answer_stream_from_hits", None)
            
            if callable(stream_fn):
                logger.info(f"callable(stream_fn)")
                last_ping = time.time()
                # 实时 token 流式输出
                async for chunk in stream_fn(
                    question, hits, decision=decision, llm_override=llm_override
                ):
                    now = time.time()
                    if now - last_ping > 1.0:
                        yield b": ping\n\n"
                        last_ping = now
                    dt = round(now - t0, 3)
                    # logger.info("[SSE] token dt=%.3f len=%d", dt, len(chunk))
                    if chunk:
                        yield _sse("token", {"text": chunk, "dt": dt})

                yield _sse("done", {"ok": True, "dt": round(time.time()-t0, 3)})

            else:
                # fallback：先生成完整答案，再手动分块发送
                final_text = "" 
                final_ans = pipeline.answer_from_hits(
                    question, hits, decision=decision, llm_override=llm_override
                )
                final_text = _get(final_ans, "answer", "") or _get(final_ans, "text", "") or ""

                chunk_size = 30
                for i in range(0, len(final_text), chunk_size):
                    yield _sse("token", final_text[i : i + chunk_size])
                    await asyncio.sleep(0.01)  # 轻微延时，防止同步阻塞事件循环

                yield _sse("done", {
                    "retrieval_id": rid or None
                })

        except Exception as e:
            logger.exception(f"RAG answer stream failed: {e}")
            yield _sse("error", {"error": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # 禁用 nginx/proxy 缓冲，确保实时推送
    }

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )

@app.post("/rag/query")
async def rag_query(body: dict, request: Request):
    """
    Single endpoint (retrieve + answer)，包括 streaming 支持。
    """
    # Stage 1: retrieve
    retrieve_resp = await rag_retrieve(body, request)
    rid = retrieve_resp.get("retrieval_id")

    if not rid:
        raise HTTPException(status_code=500, detail="Retrieval failed: no retrieval_id returned")

    # Stage 2: 构造 answer 请求，继承 stream 参数
    answer_body = {"retrieval_id": rid}

    if body.get("stream"):
        answer_body["stream"] = body["stream"]

    # if body.get("top_k") is not None:
    #     answer_body["top_k"] = body["top_k"]

    return await rag_answer(answer_body, request)

# -----------------------
# PDF ingest
# -----------------------
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    if ingest_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return await ingest_service.ingest_pdf_and_schedule(file, background_tasks)


@app.get("/ingest/status/{doc_id}")
def ingest_status(doc_id: str):
    if ingest_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return ingest_service.get_status(doc_id)
