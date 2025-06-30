from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class PathsConfig(BaseModel):
    base_dir: str = str(BASE_DIR)
    data_dir: str = str(DATA_DIR)

    raw_dir: str = str(DATA_DIR / "raw")
    processed_dir: str = str(DATA_DIR / "processed")
    index_dir: str = str(DATA_DIR / "index")
    eval_dir: str = str(DATA_DIR / "eval")
    upload_dir: str = str(DATA_DIR / "uploads")
    graph_dir: str = str(DATA_DIR / "graph")

    contract_law_raw: str = str(DATA_DIR / "raw" / "minfadian_hetongbian.txt")
    contract_law_jsonl: str = str(DATA_DIR / "processed" / "contract_law.jsonl")
    law_graph_jsonl: str = str(DATA_DIR / "graph" / "law_graph.jsonl")
    legal_kg_jsonl: str = str(DATA_DIR / "graph" / "legal_kg.jsonl")


class LLMConfig(BaseModel):
    provider: str = "qwen-local"   # "qwen-local" | "openai"
    model: str = "Qwen/Qwen2-1.5B-Instruct"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    max_context_tokens: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9

class RetrievalConfig(BaseModel):
    processed_file: str = "processed/contract_law.jsonl"
    faiss_index_file: str = "index/faiss.index"
    faiss_meta_file: str = "index/faiss_meta.jsonl"
    bm25_index_file: str = "index/bm25.pkl"

    top_k: int = 10
    bm25_weight: float = 0.4
    dense_weight: float = 0.6

    embedding_model: str = "BAAI/bge-base-zh-v1.5"  # BAAI/bge-m3 for better performance


class PDFConfig(BaseModel):
    enable_ocr: bool = True
    ocr_lang: str = "chi_sim"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class RoutingConfig(BaseModel):
    enable_router: bool = True
    llm_based: bool = False


class AppConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    pdf: PDFConfig = PDFConfig()
    server: ServerConfig = ServerConfig()
    routing: RoutingConfig = RoutingConfig()

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "AppConfig":
        if config_file:
            p = Path(config_file)
            if p.exists():
                import yaml, json
                if p.suffix in {".yml", ".yaml"}:
                    return cls(**yaml.safe_load(p.read_text(encoding="utf-8")))
                elif p.suffix == ".json":
                    return cls(**json.loads(p.read_text(encoding="utf-8")))

        cfg = cls()

        data_dir = Path(cfg.paths.data_dir)

        def abs_path(rel: str) -> str:
            return str(data_dir / rel)

        r = cfg.retrieval
        r.processed_file = abs_path(r.processed_file)
        r.faiss_index_file = abs_path(r.faiss_index_file)
        r.faiss_meta_file = abs_path(r.faiss_meta_file)
        r.bm25_index_file = abs_path(r.bm25_index_file)

        Path(cfg.paths.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.index_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.eval_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.graph_dir).mkdir(parents=True, exist_ok=True)

        return cfg
