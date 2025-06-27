from __future__ import annotations
import gradio as gr
from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline
from legalrag.pdf.parser import extract_text_from_pdf
import asyncio

cfg = AppConfig.load()
pipeline = RagPipeline(cfg)

# -----------------------
# Async RAG 接口
# -----------------------
async def rag_interface(question: str, top_k: int, answer_style: str, chat_history):
    """
    处理问题，并返回 LLM 回答、条文列表、历史对话
    """
    # 调用 async LLM
    ans = await pipeline.answer_async(question, top_k=top_k)

    # 根据风格调整回答
    response = ans["text"]
    if answer_style == "简明" and len(response) > 500:
        response = response[:500] + "..."

    # 降级模式提示
    if ans.get("mode") == "degraded":
        response = f"当前处于降级模式，LLM 未完全可用。\n\n{response}"

    # 处理条文
    hits_display = []
    for h in ans.get("hits", []):
        c = h.chunk
        hits_display.append({
            "rank": h.rank,
            "score": round(h.score, 3),
            "law_name": c.law_name,
            "chapter": c.chapter or "",
            "section": c.section or "",
            "article_no": c.article_no,
            "text": c.text,
        })

    # 更新历史记录
    chat_history = chat_history or []
    chat_history.append((question, response))

    return response, hits_display, chat_history

# -----------------------
# PDF 上传接口（同步）
# -----------------------
def ingest_pdf(file):
    """解析 PDF 并返回文本预览"""
    if not file.name.endswith(".pdf"):
        return "仅支持 PDF 文件", None, None

    tmp_path = f"{cfg.paths.upload_dir}/{file.name}"
    content = file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    
    text = extract_text_from_pdf(tmp_path, cfg)
    return f"PDF 上传成功，文本长度: {len(text)}", None, None

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

    # Async click 
    submit_btn.click(
        fn=rag_interface,
        inputs=[question_input, top_k_slider, answer_style, chat_history],
        outputs=[answer_md, hits_df, chat_history],
        show_progress=True,
    )

    pdf_upload.upload(
        fn=ingest_pdf,
        inputs=[pdf_upload],
        outputs=[answer_md, hits_df, chat_history],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, show_api=False, show_error=True)
