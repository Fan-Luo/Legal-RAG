ARG CACHE_BUST=1
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir huggingface_hub==0.36.0

COPY . /app
WORKDIR /app/data
RUN pwd && ls -la && python download_data.py

# 回到 /app 再启动服务
WORKDIR /app

RUN test -f /app/data/index/bm25.pkl \
 && test -f /app/data/index/faiss/faiss.index \
 && test -f /app/data/index/faiss/faiss_meta.jsonl


EXPOSE 7860

CMD ["bash", "-lc", "uvicorn legalrag.api.server:app --host 0.0.0.0 --port 7860 --log-level info --access-log"]
