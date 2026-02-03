

**法律条文检索、问答与推理系统**

[![HuggingFace Spaces](https://img.shields.io/badge/Space-Legal--RAG-blue?logo=huggingface)](https://huggingface.co/spaces/flora-l/Legal-RAG)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/fanlcs/retrieval-performance-evaluation)
[![Colab Notebook](https://img.shields.io/badge/Run-Colab-blue?logo=googlecolab)](https://colab.research.google.com/drive/1TRp4d_VwlcSY8f78psuCNX_90WA3g6qS?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

[Legal-RAG](https://fan-luo.github.io/Legal-RAG/) 以法律文本为核心语料，完成条文预处理、BM25/FAISS/ColBERT 索引与 Graph 的构建，并通过 FastAPI 提供检索与问答服务。OpenAI 作为生成模型是可选的，可以在有 GPU 的环境中启用本地模型（默认：Qwen），也可通过配置切换其他开源模型。

<video src="docs/project.mp4" width="720" height="480" controls muted style="display: block; margin: 0 auto 50px auto;"></video>

## 在线演示  
- 1 [Hugging Face Spaces](https://huggingface.co/spaces/flora-l/Legal-RAG)（未启用 GPU，较慢，需要 OpenAI key）
- 2 [Colab notebook](https://colab.research.google.com/drive/1bDlIFzHvnlR-U3lWVGLJAGq3KwcpvlxG?usp=sharing)（在GPU上启动服务，不需要 OpenAI key）

<a class="github-video">https://github.com/user-attachments/assets/1a380d62-d909-480a-8618-a03f3015e1bd</a>


## 功能特性 

* 法条预处理：原始文本清洗、分条、结构化为 JSONL
* 路由：
  - 规则路由识别任务/问题类型
  - 可选 LLM 路由覆盖 
* 混合检索：
  - BM25（关键词匹配）
  - Dense（BGE + FAISS 向量检索）
  - ColBERT（late interaction 精排通道）
  - 多通道加权融合
  - 按语言路由（zh/en）
  - RRF + weighted sum 融合多通道结果
  - Graph（基于法条关系扩展候选）
  - Reranker（CrossEncoder 或 LLM 重排）
* LLM 生成：
  - 本地 Qwen（默认）
  - 可选 OpenAI 兼容接口
* 服务与界面：
  - FastAPI 后端 API
  - Web UI（多轮问答、条文展示、PDF 上传）
* 脚本支持：
  - 预处理 / 索引构建 / 检索评估
 

## 系统架构 

<img src="docs/architecture.png" alt="Legal-RAG Architecture" width="800"/>


## 快速开始 


### 1. 克隆项目并安装依赖
```bash
git clone https://github.com/Fan-Luo/Legal-RAG.git
cd Legal-RAG
pip install -r requirements.txt
````

### 2. 准备法律数据并构建索引
法律数据位于 `data/raw/`，可替换或增加需要的其他法律文本。
默认语料包含：

- 中文：中华人民共和国民法典
- 英文：Uniform Commercial Code（UCC）

系统会根据问题语言路由到对应语料与索引。

```bash

# 预处理为 JSONL
python -m scripts.preprocess_law

# 构建 FAISS + BM25 + ColBERT 索引
python -m scripts.build_index

# 构建 法律知识图谱
python -m scripts.build_graph
````
生成的语料按语言拆分：

- `data/processed/law_zh.jsonl`, `data/processed/law_en.jsonl`
- `data/index/zh/...`, `data/index/en/...`
- `data/graph/law_graph_zh.jsonl`, `data/graph/law_graph_en.jsonl`

### 3. 启动 API 服务
```bash
python -m uvicorn legalrag.api.server:app --host 127.0.0.1 --port 8000
````
> 默认使用本地 Qwen 模型（如 Qwen/Qwen2.5-3B-Instruct），需要本地机器有GPU 和足够内存。
> 如需使用 OpenAI，请参考 「在线演示（Hugging Face Spaces） / OpenAI 配置」。

### 4. 打开演示界面
访问：http://127.0.0.1:8000/ 或 http://127.0.0.1:8000/ui/


提示：

- PDF 上传建议使用可复制文本的 PDF，以提高解析准确率。
- 本地 LLM（Qwen / BGE）建议使用 GPU 或充足显存；OpenAI API 为可选方案。

## 服务拆分 
- API 服务：`legalrag.api.server:app`
- 检索服务：`legalrag.services.retrieval_api:app`
- 索引管理服务：`legalrag.services.index_api:app`

本地多服务启动：
```bash
docker compose up --build
```

索引版本管理：
```bash
python scripts/build_index.py --index-version v1 --activate
python scripts/index_admin.py list
python scripts/index_admin.py activate v1
```


## 示例 

```python
from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline

cfg = AppConfig.load(None)
pipeline = RagPipeline(cfg)

question = "业主大会的决定有法律效应吗？"
ans = pipeline.answer(question)

print("Question:", question)
print("Answer:", ans.answer)
```
 

## 项目结构 

```
Legal-RAG/
│
├── legalrag/
│   ├── __init__.py
│   ├── config.py                   
│   ├── schemas.py                 
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── gateway.py
│   │   └── client.py              # Qwen / OpenAI LLMClient 
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── legal_issue_extractor.py
│   │   └── router.py              # QueryType + Graph/RAG Suggestions
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── builders
│   │   ├── dense_retriever.py     # Dense (BGE + FAISS)
│   │   ├── vector_store.py        
│   │   ├── bm25_retriever.py      # Sparse (BM25 + jieba)
│   │   ├── colbert_retriever.py
│   │   ├── hybrid_retriever.py    # Dense + Sparse + Colbert + Graph + Rerank
│   │   ├── by_lang_retriever.py   # zh/en routing
│   │   ├── corpus_loader.py       # read all chunks from processed_dir
│   │   ├── incremental_indexer.py
│   │   ├── graph_retriever.py
│   │   ├── graph_store.py         # law_graph / legal_kg  
│   │   └── rerankers.py
│   │
│   ├── pdf/
│   │   ├── __init__.py
│   │   └── parser.py              # pdfplumber + OCR fallback
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── ingestor.py            # LawChunk JSONL 
│   │   ├── orchestrator.py        # jobs + status
│   │   ├── task_queue.py
│   │   └── service.py         
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py        # Graph-aware RAG 核心推理链
│   │
│   ├── prompts/
│   │   ├──  prompt_en.json        # English Prompt 
│   │   └──  prompt_zh.json        # Chinese Prompt 
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── lang.py
│   │   ├── logger.py              # 日志 
│   │   └── text.py                # 文本清洗 / 正则工具
│   │
│   └── api/
│       ├── __init__.py
│       └── server.py              # FastAPI（/rag/retrieve, /rag/answer, /ingest/pdf）
│
├── ui/
│   ├── index.html
│   └── demo.qmd
│
├── scripts/
│   ├── preprocess_law.py          # 法条解析 → LawChunk JSONL
│   ├── build_index.py             # FAISS + BM25 索引构建
│   ├── build_graph.py             # law_graph / legal_kg 构建
│   ├── bgenerate_synthetic_data.py
│   └── evaluate_retrieval.py      # Hit@K / MRR / nDCG
│
├── notebooks/
│   ├── 01_Launch_the_UI.ipynb
│   ├── 02_LegalRAG_Pipeline.ipynb
│   ├── 03_Retrieval_Performance_Evaluation.ipynb
│   └── 04_Law_Graph_Visualization.ipynb
│
├── data/
│   ├── raw/                         
│   │   ├── minfadian.txt            
│   │   └── ucc/                    
│   ├── processed/                 # law_zh.jsonl / law_en.jsonl
│   ├── index/                     # faiss/bm25/colbert per language
│   └── graph/                     # law_graph_zh.jsonl / law_graph_en.jsonl
│   └── eval/
│
├── docs/
│   ├── architecture.mmd
│   └── architecture.png
│
├── tests/
│   ├── test_router.py
│   └── test_retrieval.py
│
├── README.md
├── README-zh.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── _quarto.yml
├── index.qmd
├── app.py                           # Hugging Face Space 入口
├── Dockerfile
└── .gitignore                       
               

```


## 可扩展方向 
  - 更复杂的法律知识图谱
  - 高级查询路由 / 多模型融合
  - 支持其他法律领域或多语种

## 许可声明 

Apache License 2.0

本仓库仅包含源码，不包含第三方模型权重。用户需自行遵守所使用模型的许可证（如 Qwen、BGE、OpenAI 等）。


## 免责声明 

本项目仅用于提供法律信息辅助，供学习与研究参考之用。
使用者不应将本项目作为专业法律咨询的替代，因使用本项目所产生的任何直接或间接后果，项目作者及贡献者不承担任何责任。
