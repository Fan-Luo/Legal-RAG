# Legal-RAG

This repository is a Chinese Contract Law (Civil Code – Contract Part) Retrieval-Augmented Generation (RAG) system.

It includes:

- Law preprocessing (Civil Code – Contract Part) → JSONL
- Hybrid retrieval (FAISS + BM25)
- Qwen2-1.5B-Instruct local LLM (preferred) and optional OpenAI API
- Simple law-graph store and query router
- FastAPI backend and Gradio web UI
- Scripts for preprocessing, index building and basic retrieval evaluation

## Quick Start (Local)

```bash
pip install -r requirements.txt

# 1) Put the raw contract law text into: data/raw/minfadian_hetongbian.txt

python -m scripts.preprocess_law
python -m scripts.build_index

# 2) Run API
uvicorn legalrag.api.server:app --port 8000

# 3) Run Gradio demo
python ui/gradio_app.py
```

## Notes

This repo is a working skeleton and can be extended to:
- richer law_graph construction
- more advanced routing
- different laws
