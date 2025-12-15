---
title: Legal-RAG
emoji: "🤖"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: true
---

# Legal-RAG


[![HuggingFace Spaces](https://img.shields.io/badge/Space-Legal--RAG-blue?logo=huggingface)](https://huggingface.co/spaces/flora-l/Legal-RAG)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue)]
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> **Contract Law Retrieval-Augmented Generation (RAG) system**  
A production-grade Retrieval-Augmented Generation (RAG) system for Chinese Contract Law (《民法典·合同编》)
Focused on correctness, traceability, and engineering clarity — not legal advice.

 

## What is Legal-RAG?
Legal-RAG is an open-source, end-to-end legal information retrieval and reasoning system designed around the Chinese Civil Code – Contract Book.

It demonstrates how to build a law-aware RAG system that is:
- grounded in explicit statutory text
- engineered with retrieval transparency
- structured for future extensibility (graph / routing / SaaS-ready)


## 🤗 Hugging Face Spaces Demo (Online)

This project provides a fully functional online demo deployed on Hugging Face Spaces.
### Live Demo
👉 https://huggingface.co/spaces/flora-l/Legal-RAG 

In the Hugging Face Space:
  **Settings → Variables and secrets**

  Set:
  
    - OPENAI_API_KEY (required)
    - OPENAI_MODEL (optional, e.g. gpt-4o-mini)

## Features

### Law-aware RAG 
- Explicit article-level chunking
- Law-specific metadata (chapter / section / article number)
- Retrieval results are inspectable and auditable

### Hybrid Retrieval, Done Properly
- Dense retrieval: FAISS
- Sparse retrieval: BM25
- Weighted fusion via HybridRetriever 

### Query Routing & Graph Awareness
- Lightweight law_graph for structural reasoning
- Router decides between:
  - pure retrieval
  - graph-assisted RAG
- Clear extension point for richer legal knowledge graphs

### Online PDF Ingestion (Incremental Indexing)
- Upload PDFs → parse → chunk → JSONL
- Incremental FAISS add
- BM25 rebuild in background

### Engineering-first Design
- Clear module boundaries
- Deterministic data flow
- Minimal magic, maximal readability
- SaaS-compatible architecture without being a SaaS

 

## System Architecture
The system is organized into four clearly separated layers:

1. Offline Build
  Law text preprocessing, index construction, graph building

2. Index Artifacts
  FAISS, BM25, and law_graph as immutable read models

3. Online Ingestion
  PDF upload → background incremental indexing

4. Online Serving (RAG + Routing)
  FastAPI + RagPipeline + Router + LLM

See the architecture diagram for the full data flow.

<img src="docs/architecture.png" alt="Legal-RAG Architecture" width="800"/>

 
## Quickstart (Local)
### 1. Clone & install
```bash
git clone https://github.com/Fan-Luo/Legal-RAG.git
cd Legal-RAG
pip install -r requirements.txt
````

### 2.Prepare law data & build index
```bash
# preprocess law text into structured JSONL
python -m scripts.preprocess_law

# build FAISS + BM25 indexes
python -m scripts.build_index

# build law_graph
python -m scripts.build_graph
````
### 3. Start API service
```bash
python -m uvicorn legalrag.api.server:app --host 127.0.0.1 --port 8000 
````

### 4. Launch Demo
visit http://127.0.0.1:8000/ or http://127.0.0.1:8000/ui/
 

## Example

```python
from legalrag.config import AppConfig
from legalrag.pipeline.rag_pipeline import RagPipeline

cfg = AppConfig.load()
pipeline = RagPipeline(cfg)

question = "合同生效后，如果对价款和履行地点没有约定，应当如何处理？"
ans = pipeline.answer(question)

print(ans.answer)
```

> 1. 结论：
>   - 经过全面分析与理解，我们认为，当合同对价款和履行地点没有约定时，合同生效后，当事人可以根据合同相关条款或者交易习惯确定支付价款和履行地点。这体现了合同自由的原则和诚实信用的基本精神。
>   
> 2. 分析与理由：
>   - 我们首先确认了《民法典·合同编》第五百一十条中明确规定的合同生效后当事人的支付地点选择权：
>     - 在没有具体约定的情况下，应由双方协商确定或依据合同惯例；
>     - 如协商不成，可依合同相关条款或交易习惯确定。
>     
>   - 对于履行地点的选择，我们援引了第六百二十七条中的相关规定，强调了在合同签订时就已经明确了合同履行地点。尽管如此，这一条款并不足以涵盖所有可能的情况，因此我们还需要考虑合同的实际履行情况来进一步判断。
>
> 3. 参考条文列表：
>   - （核心依据）
>       - 第五百一十条
>   - （次要参考）
>       - 第六百二十七条

 


## LLM Backends & Cost Model
Supported backends:
- Local LLM (Qwen series, need GPU and enough memory)
- OpenAI-compatible API (need to provide OpenAI API key)

Important design choice

- No API key is collected via UI
- LLM keys are read only from environment variables
- If no key is provided and no local model loaded, the system gracefully degrades


## 📂 Project Structure

```
Legal-RAG/
│
├── legalrag/
│   ├── __init__.py
│   ├── config.py                  # AppConfig / Paths / LLM / Retrieval
│   ├── models.py                  # LawChunk / RetrievalHit / RoutingDecision / RagAnswer
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py              # Qwen / OpenAI LLMClient（async-safe）
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # Dense (BGE + FAISS)
│   │   ├── bm25_retriever.py      # Sparse (BM25 + jieba)
│   │   ├── hybrid_retriever.py    # Dense + Sparse + 权重融合
│   │   ├── corpus_loader.py       # read all chunks from processed_dir
│   │   ├── incremental_indexer.py
│   │   └── graph_store.py         # law_graph / legal_kg 读取与 walk
│   │
│   ├── routing/
│   │   ├── __init__.py
│   │   └── router.py              # QueryType + Graph/RAG Suggestions
│   │
│   ├── pdf/
│   │   ├── __init__.py
│   │   └── parser.py              # pdfplumber + OCR fallback
│   │
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── ingestor.py            # PDFIngestor 
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py        # Graph-aware RAG Core Inference
│   │
│   ├── prompts/
│   │   └── legal_rag_prompt.txt   # Prompt 
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py             
│   │   └── text.py                
│   │
│   └── api/
│       ├── __init__.py
│       └── server.py              # FastAPI（/rag/query, /, ingest/pdf）
│
├── ui/
│   └── index.html
│
├── scripts/
│   ├── preprocess_law.py          # parse law → LawChunk JSONL
│   ├── build_index.py             # FAISS + BM25 indexes
│   ├── build_graph.py             # law_graph / legal_kg  
│   └── evaluate_retrieval.py      # Hit@K / MRR / nDCG
│
├── notebooks/
│   ├── 01_kaggle_build_index_and_eval.ipynb
│   ├── 02_colab_qwen_rag_demo.ipynb
│   ├── 03_retrieval_visualization.ipynb
│   ├── 04_retrieval_benchmark_legal.ipynb
│   └── 05_rag_answer_eval.ipynb
│
├── data/
│   ├── raw/                         
│   │   └── minfadian.txt            
│   └── eval/
│       └── contract_law_qa.jsonl
├── docs/
│   ├── architecture.mmd
│   └── architecture.png
├── tests/
│   ├── test_router.py
│   └── test_retrieval.py
├── README.md
├── README-zh.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── app.py                           # Hugging Face Spaces entry
├── Dockerfile
└── .gitignore                       

```

 

## Extensibility

Legal-RAG is intentionally structured to support:

- richer legal knowledge graphs
- multi-document reasoning
- multi-tenant isolation
- BYOK (Bring Your Own Key) SaaS models

These are architectural affordances, not product promises.



## Who is this project for?
This repository is intended for:
- Engineers exploring RAG system design
- Researchers working on legal NLP / AI + law
- Practitioners interested in traceable AI systems
- Candidates demonstrating architecture-level thinking

> ⚠️ This project provides legal information assistance for educational and research purposes only and does not constitute legal advice. Users should not rely on this project as a substitute for professional legal counsel. The authors and contributors disclaim any liability for any direct or indirect consequences arising from the use of this project.




## License
Apache License 2.0

This repository contains source code only.
Users are responsible for complying with the licenses of any models or APIs they choose to integrate.