ARG CACHE_BUST=1
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
WORKDIR /app/data
RUN pwd && ls -la && python download_data.py

# 回到 /app 再启动服务
WORKDIR /app

RUN test -f /app/data/index/zh/bm25.pkl \
 && test -f /app/data/index/zh/faiss/faiss.index \
 && test -f /app/data/index/zh/faiss/faiss_meta.jsonl


EXPOSE 7860

CMD ["bash", "-lc", "uvicorn legalrag.api.server:app --host 0.0.0.0 --port 7860 --log-level info --access-log"]
