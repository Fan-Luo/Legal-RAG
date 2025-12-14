FROM python:3.10-slim

WORKDIR /app

# system deps（faiss / torch 常用）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .
COPY ui /app/ui

WORKDIR /app
COPY . /app


# expose FastAPI port
EXPOSE 8000

# 启动 FastAPI
CMD ["uvicorn", "legalrag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]