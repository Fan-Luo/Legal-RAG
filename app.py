from __future__ import annotations
import os
import re
import json
import time
import asyncio
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from legalrag.config import AppConfig
from legalrag.ingest.ingestor import PDFIngestor
from legalrag.retrieval.incremental_indexer import IncrementalIndexer
from legalrag.retrieval.bm25_retriever import BM25Retriever
from legalrag.pipeline.rag_pipeline import RagPipeline

# -----------------------------
# Paths / Config for Spaces
# -----------------------------
# Spaces filesystem is writable under /home/user/app and /tmp
ROOT = Path(os.environ.get("SPACE_WORKDIR", "/home/user/app")).resolve()
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

for p in [UPLOAD_DIR, PROCESSED_DIR, INDEX_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def load_cfg() -> AppConfig:
    cfg = AppConfig.load()

    # Ensure all runtime paths are writable in Spaces
    cfg.paths.upload_dir = str(UPLOAD_DIR)
    cfg.paths.processed_dir = str(PROCESSED_DIR)

    # Put indexes under data/index
    cfg.retrieval.faiss_index_file = str(INDEX_DIR / "faiss.index")
    cfg.retrieval.faiss_meta_file = str(INDEX_DIR / "faiss_meta.jsonl")
    cfg.retrieval.bm25_index_file = str(INDEX_DIR / "bm25.pkl")

    return cfg


CFG = load_cfg()

# Lazily init pipeline (avoid heavy load at import time)
_PIPELINE_LOCK = threading.Lock()
_PIPELINE: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = RagPipeline(CFG)
    return _PIPELINE


# -----------------------------
# Post-processor
# -----------------------------
_CONC_RE = re.compile(r"\b1\.\s*结论[:：]")


def trim_to_conclusion(raw_answer: str) -> str:
    """Remove everything before '1. 结论' (keep from it)."""
    if not raw_answer:
        return raw_answer
    m = _CONC_RE.search(raw_answer)
    if not m:
        return raw_answer.strip()
    return raw_answer[m.start():].strip()


# -----------------------------
# Background jobs + status
# -----------------------------
EXEC = ThreadPoolExecutor(max_workers=2)
STATUS_LOCK = threading.Lock()
INGEST_STATUS: dict[str, dict] = {}  # doc_id -> status dict


def _set_status(doc_id: str, **kwargs):
    with STATUS_LOCK:
        s = INGEST_STATUS.setdefault(doc_id, {})
        s.update(kwargs)


def _get_status(doc_id: str) -> dict:
    with STATUS_LOCK:
        return dict(INGEST_STATUS.get(doc_id, {}))


def faiss_index_job(doc_id: str, jsonl_path: str):
    try:
        _set_status(doc_id, faiss="running", error=None)
        added = IncrementalIndexer(CFG).add_jsonl(jsonl_path)
        _set_status(doc_id, faiss="done", added=int(added))
    except Exception as e:
        _set_status(doc_id, faiss="failed", error=str(e))


def bm25_rebuild_job(doc_id: str):
    try:
        _set_status(doc_id, bm25="running", error=None)
        BM25Retriever(CFG).build()
        _set_status(doc_id, bm25="done")
    except Exception as e:
        _set_status(doc_id, bm25="failed", error=str(e))


# -----------------------------
# Gradio handlers
# -----------------------------
def ingest_pdf(file) -> tuple[str, str, bool]:
    """
    Upload handler:
    - save file
    - ingest -> writes jsonl and returns preview/doc_id/jsonl_path
    - schedule background FAISS add + BM25 rebuild
    Returns: (ingest_md, doc_id_state, timer_active)
    """
    if file is None:
        return "请先选择一个 PDF 文件。", "", False

    # Gradio file provides a local temp path at file.name
    src_path = Path(getattr(file, "name", "")).resolve()
    if not src_path.exists() or src_path.suffix.lower() != ".pdf":
        return "仅支持上传 PDF 文件。", "", False

    # Copy to our upload dir (optional but keeps things organized)
    dst_path = UPLOAD_DIR / src_path.name
    try:
        dst_path.write_bytes(src_path.read_bytes())
    except Exception as e:
        return f"保存上传文件失败：{e}", "", False

    # Run ingestion (may be slow for OCR PDFs)
    try:
        ingestor = PDFIngestor(CFG)
        result = ingestor.ingest_pdf_to_jsonl(dst_path, law_name=f"用户上传PDF-{dst_path.name}")

        doc_id = result.doc_id
        _set_status(doc_id, faiss="scheduled", bm25="scheduled", added=0, error=None)

        # Schedule background jobs
        EXEC.submit(faiss_index_job, doc_id, result.jsonl_path)
        EXEC.submit(bm25_rebuild_job, doc_id)

        md = (
            f"### 已解析\n"
            f"- doc_id: `{doc_id}`\n"
            f"- chunks: `{result.num_chunks}`\n"
            f"- jsonl: `{result.jsonl_path}`\n\n"
            f"### 预览\n"
            f"{result.preview}\n\n"
            f"### 索引状态\n"
            f"- FAISS / BM25 已提交后台任务（每 2 秒自动刷新）"
        )
        return md, doc_id, True

    except Exception as e:
        return f"PDF 解析失败：{e}", "", False


def poll_status(doc_id: str) -> tuple[str, bool]:
    """
    Timer tick: read status for doc_id.
    Returns: (status_md, timer_active)
    """
    if not doc_id:
        return "索引状态：-", False

    s = _get_status(doc_id)
    if not s:
        return "索引状态：未找到 doc_id（可能已重启或尚未写入状态）", False

    faiss = s.get("faiss", "unknown")
    bm25 = s.get("bm25", "unknown")
    added = s.get("added", 0)
    err = s.get("error")

    done = (faiss == "done") and (bm25 == "done")
    failed = (faiss == "failed") or (bm25 == "failed")

    text = f"索引状态：FAISS={faiss}（added={added}），BM25={bm25}"
    if err:
        text += f"\n错误：{err}"

    # stop polling once done/failed
    return text, (not (done or failed))


def _run_pipeline_answer(question: str, top_k: int, answer_style: str):
    """
    Calls pipeline in a robust way: supports either async or sync API.
    Expected to return a dict-like object with 'text' and optional 'hits'.
    """
    pipeline = get_pipeline()

    # Prefer async method if exists
    if hasattr(pipeline, "answer_async"):
        return asyncio.run(pipeline.answer_async(question, top_k=int(top_k), answer_style=answer_style))
    # Fallback to sync
    return pipeline.answer(question, top_k=int(top_k), answer_style=answer_style)


def rag_ask(question: str, top_k: int, answer_style: str, history: list):
    question = (question or "").strip()
    if not question:
        return "请先输入问题。", [], history

    try:
        ans = _run_pipeline_answer(question, top_k, answer_style)

        # normalize answer text
        text = ans["text"] if isinstance(ans, dict) and "text" in ans else str(ans)
        text = trim_to_conclusion(text)

        # hits table (best-effort)
        rows = []
        hits = ans.get("hits", []) if isinstance(ans, dict) else []
        for h in hits:
            # tolerate different hit schemas
            chunk = getattr(h, "chunk", None) or (h.get("chunk") if isinstance(h, dict) else None)
            score = getattr(h, "score", None) if not isinstance(h, dict) else h.get("score", None)
            rank = getattr(h, "rank", None) if not isinstance(h, dict) else h.get("rank", None)

            if chunk is not None:
                law_name = getattr(chunk, "law_name", "") if not isinstance(chunk, dict) else chunk.get("law_name", "")
                chapter = getattr(chunk, "chapter", "") if not isinstance(chunk, dict) else chunk.get("chapter", "")
                section = getattr(chunk, "section", "") if not isinstance(chunk, dict) else chunk.get("section", "")
                article_no = getattr(chunk, "article_no", "") if not isinstance(chunk, dict) else chunk.get("article_no", "")
                ctext = getattr(chunk, "text", "") if not isinstance(chunk, dict) else chunk.get("text", "")
            else:
                law_name = chapter = section = article_no = ""
                ctext = ""

            rows.append([
                rank if rank is not None else "",
                float(score) if score is not None else 0.0,
                law_name,
                chapter or "",
                section or "",
                article_no or "",
                (ctext or "")[:400],
            ])

        history = history or []
        history.append((question, text))
        return text, rows, history

    except Exception as e:
        return f"查询失败：{e}", [], history


# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Legal-RAG (HF Spaces)")
    gr.Markdown("上传 PDF → 自动解析并后台增量索引；随后可进行 RAG 问答。")

    doc_id_state = gr.State(value="")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_upload = gr.File(label="上传 PDF", file_types=[".pdf"])
            ingest_md = gr.Markdown("上传后会显示解析预览。")
            status_md = gr.Markdown("索引状态：-")

        with gr.Column(scale=2):
            question = gr.Textbox(label="请输入法律问题", lines=3, placeholder="例如：合同生效后，价款和履行地点未约定怎么办？")
            top_k = gr.Slider(3, 20, value=9, step=1, label="检索条文数 (top_k)")
            answer_style = gr.Dropdown(["专业", "通俗", "简洁"], value="专业", label="回答风格")
            ask_btn = gr.Button("提问")
            answer_md = gr.Markdown()
            hits_df = gr.Dataframe(
                headers=["rank", "score", "law_name", "chapter", "section", "article_no", "text"],
                datatype=["str", "number", "str", "str", "str", "str", "str"],
                row_count=10,
                col_count=7,
                wrap=True,
                interactive=False,
                label="命中条文（Top-K）",
            )
            chat_history = gr.State(value=[])

    timer = gr.Timer(2, active=False)

    pdf_upload.upload(
        fn=ingest_pdf,
        inputs=[pdf_upload],
        outputs=[ingest_md, doc_id_state, timer],
        show_progress=True,
    )

    timer.tick(
        fn=poll_status,
        inputs=[doc_id_state],
        outputs=[status_md, timer],
    )

    ask_btn.click(
        fn=rag_ask,
        inputs=[question, top_k, answer_style, chat_history],
        outputs=[answer_md, hits_df, chat_history],
        show_progress=True,
    )

demo.queue(concurrency_count=2).launch(server_name="0.0.0.0", server_port=7860)
