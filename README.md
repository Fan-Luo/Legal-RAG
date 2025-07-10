# Legal-RAG

This repository is a Chinese Contract Law (Civil Code – Contract Part) Retrieval-Augmented Generation (RAG) system.

It includes:

- Law preprocessing (Civil Code – Contract Part) → JSONL
- Hybrid retrieval (FAISS + BM25)
- Qwen2-1.5B-Instruct local LLM (preferred) and optional OpenAI API
- Simple law-graph store and query router
- FastAPI backend and Gradio web UI
- Scripts for preprocessing, index building and basic retrieval evaluation

## License

This project is licensed under the Apache License 2.0.

This repository contains source code only and does not redistribute any
third-party model weights. Users are responsible for complying with the
licenses of models they choose to load (e.g. Qwen, BGE, OpenAI).


**Legal-RAG** 是一个针对中国《民法典·合同编》的 **检索增强生成（RAG, Retrieval-Augmented Generation）系统**，可实现基于条文的问答、条文检索和多轮对话。


## 功能概览

- **法律条文预处理**：将《民法典·合同编》文本转成 JSONL 格式，方便索引。
- **混合检索**：FAISS 向量检索 + BM25 精确匹配，支持 top-k 条文检索。
- **LLM 回答**：
  - 本地 Qwen2-1.5B-Instruct（推荐）
  - 可选 OpenAI API
- **法律知识图谱与路由**：简单的 law-graph 存储与查询路由。
- **接口与 UI**：
  - FastAPI 后端服务
  - Gradio Web UI，支持多轮问答与条文展示
- **辅助脚本**：
  - 预处理、索引构建、检索评估



## 安装与依赖

```
git clone <repo-url>
cd legalrag
pip install -r requirements.txt
```

## 快速开始 
1. 数据准备与索引构建
```
# 将原始《民法典·合同编》文本放到：
data/raw/minfadian_hetongbian.txt

# 生成 JSONL 数据
python -m scripts.preprocess_law

# 构建检索索引
python -m scripts.build_index
```
2. 启动 API 服务
```
uvicorn legalrag.api.server:app --host 0.0.0.0 --port 8000
```
API 默认提供：

  - /health 健康检查

  - /rag/query 法律问题问答

  - /ingest/pdf PDF 文档上传与解析

3. 启动 Gradio Demo
``` 
python ui/gradio_app.py

```
Gradio Demo 支持：
多轮对话

条文折叠展示

PDF 上传解析

回答风格选择（专业 / 简明 / 详细）

## 提示
- 可进一步扩展：
  - 更丰富的法律知识图谱，
  - 高级查询路由和多模型融合，
  - 支持其他法律或多语种数据。
- 对PDF上传，建议保证文本清晰，以提高解析准确性
- 本地 LLM 需要足够显存或 GPU 支持，OpenAI API 为可选方案
