from __future__ import annotations
import gradio as gr
from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline
import asyncio
from legalrag.ingest.pdf_ingestor import PDFIngestor
from legalrag.retrieval.incremental_indexer import IncrementalIndexer
from pathlib import Path
import requests
import os

cfg = AppConfig.load()
pipeline = RagPipeline(cfg)
API_BASE = os.environ.get("LEGALRAG_API_BASE", "http://127.0.0.1:8000")

def rag_interface(question: str, top_k: int, answer_style: str, history: list):
    question = (question or "").strip()
    if not question:
        return "请先输入问题。", [], history

    try:
        payload = {
            "question": question,
            "top_k": int(top_k),
            "answer_style": answer_style,
        }
        resp = requests.post(f"{API_BASE}/rag/query", json=payload, timeout=120)
        if resp.status_code != 200:
            return f"查询失败：{resp.status_code} {resp.text}", [], history

        data = resp.json()
        answer_text = data.get("answer", "")

        # hits -> dataframe rows 
        rows = []
        for h in data.get("hits", []):
            rows.append([
                h.get("rank", ""),
                round(float(h.get("score", 0.0)), 4),
                h.get("law_name", ""),
                h.get("chapter", ""),
                h.get("section", ""),
                h.get("article_no", ""),
                (h.get("text", "") or "")[:500],
            ])

        history = history or []
        history.append((question, answer_text))
        return answer_text, rows, history

    except Exception as e:
        return f"查询异常：{e}", [], history

# -----------------------
# PDF 上传接口 
# -----------------------

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

def poll_index_status(doc_id: str):
    if not doc_id:
        return "索引状态：-", False  # 无 doc_id，不轮询

    try:
        r = requests.get(f"{API_BASE}/ingest/status/{doc_id}", timeout=10)
        data = r.json() if r.status_code == 200 else {"error": r.text}

        if data.get("error"):
            return f"索引状态：{data.get('error')}", False

        faiss = data.get("faiss", "unknown")
        bm25 = data.get("bm25", "unknown")
        added = data.get("added", 0)

        text = f"索引状态：FAISS {faiss}（added={added}）；BM25 {bm25}"
        done = (faiss == "done") and (bm25 == "done")
        failed = (faiss == "failed") or (bm25 == "failed")

        # done/failed 都停止轮询
        return text + (f"\n错误：{data.get('error')}" if failed and data.get("error") else ""), (not (done or failed))

    except Exception as e:
        return f"索引状态：轮询异常：{e}", False

def ingest_pdf(file):
    """
    Gradio upload handler: send PDF to FastAPI /ingest/pdf.
    Outputs: (answer_md, hits_df, chat_history)
    """
    if file is None:
        return "请先选择一个 PDF 文件。", [], []

    contents = file.read()
    size = len(contents)
    if size > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"文件过大 （{size/1024/1024:.2f} MB），最大仅支持 {MAX_UPLOAD_SIZE}MB")


    # Gradio File 对象通常有 .name（本地临时路径）
    path = getattr(file, "name", None)
    if not path or not str(path).endswith(".pdf"):
        return "仅支持 PDF 文件", [], []

    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "application/pdf")}
            resp = requests.post(f"{API_BASE}/ingest/pdf", files=files, timeout=120)
        if resp.status_code != 200:
            return f"上传失败：{resp.status_code} {resp.text}", [], []

        data = resp.json()
        doc_id = data.get("doc_id", "")
        #  FastAPI 的返回字段：text_preview / doc_id / num_chunks / jsonl_path / indexed_chunks
        msg = (
            f"PDF 已解析（doc_id={doc_id}），共 {data.get('num_chunks')} 个片段。\n\n"
            f"预览：\n{data.get('text_preview', '')}"
        )

        status = "索引状态：FAISS scheduled；BM25 scheduled（后台进行中）"
        return msg, [], [], doc_id, status, True


    except Exception as e:
        return f"上传异常：{e}", [], []

# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# Legal-RAG：民法典合同编 咨询顾问")
    gr.Markdown("左侧输入问题、选择检索条文数量和回答风格，右侧显示回答和条文详情。")

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="请输入法律问题", lines=4, placeholder="如：合同违约金是否合理？")
            top_k_slider = gr.Slider(3, 20, value=10, step=1, label="检索条文数 (top_k)")
            answer_style = gr.Dropdown(choices=["专业", "简明", "详细"], value="专业", label="回答风格")
            pdf_upload = gr.File(file_types=[".pdf"], label="上传 PDF 文档")
            submit_btn = gr.Button("生成回答", variant="primary")

        with gr.Column(scale=3):
            answer_md = gr.Markdown(label="LLM 回答", interactive=False)
            with gr.Accordion("检索到的条文详情", open=False):
                hits_df = gr.Dataframe(
                    headers=["rank","score","law_name","chapter","section","article_no","text"],
                    interactive=False,
                    max_rows=10,
                    col_widths=[50,60,120,80,80,80,400],
                    datatype=["number","number","str","str","str","str","str"],
                    overflow="wrap",
                )
            chat_history = gr.Chatbot(label="历史对话记录")
            doc_id_state = gr.State(value="")
            index_status_md = gr.Markdown("索引状态：-", interactive=False)
            timer = gr.Timer(2, active=False)  # 默认不轮询


    submit_btn.click(
        fn=rag_interface,
        inputs=[question_input, top_k_slider, answer_style, chat_history],
        outputs=[answer_md, hits_df, chat_history],
        show_progress=True,
    )

    pdf_upload.upload(
        fn=ingest_pdf,
        inputs=[pdf_upload],
        outputs=[answer_md, hits_df, chat_history, doc_id_state, index_status_md, timer],
        show_progress=True,
    )

    timer.tick(
        fn=poll_index_status,
        inputs=[doc_id_state],
        outputs=[index_status_md, timer],  # 第二个输出控制 timer.active
    )
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, show_api=False, show_error=True)
