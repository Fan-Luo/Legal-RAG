from __future__ import annotations

import gradio as gr

from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline

cfg = AppConfig.load()
pipeline = RagPipeline(cfg)


def rag_interface(question: str, top_k: int):
    ans = pipeline.answer(question, top_k=top_k)
    hits_display = []
    for h in ans.hits:
        c = h.chunk
        header = f"{c.law_name} {c.article_no}"
        if c.chapter:
            header = f"{c.chapter} - {header}"
        if c.section:
            header = f"{c.section} - {header}"
        hits_display.append(
            {
                "rank": h.rank,
                "score": round(h.score, 3),
                "law_name": c.law_name,
                "chapter": c.chapter or "",
                "section": c.section or "",
                "article_no": c.article_no,
                "text": c.text,
            }
        )
    return ans.answer, hits_display


with gr.Blocks() as demo:
    gr.Markdown("# Legal-RAG：民法典合同编 RAG Demo")

    with gr.Row():
        with gr.Column(scale=2):
            q = gr.Textbox(
                label="请输入法律问题",
                value="合同约定的违约金为合同金额的 40%，是否合理？",
                lines=4,
            )
            top_k = gr.Slider(3, 20, value=10, step=1, label="检索条文数 (top_k)")
            btn = gr.Button("检索并生成回答")

        with gr.Column(scale=3):
            answer = gr.Markdown(label="LLM 回答")

    hits = gr.Dataframe(
        headers=[
            "rank",
            "score",
            "law_name",
            "chapter",
            "section",
            "article_no",
            "text",
        ],
        label="检索到的条文",
        interactive=False,
    )

    btn.click(rag_interface, inputs=[q, top_k], outputs=[answer, hits])

if __name__ == "__main__":
    demo.launch()
