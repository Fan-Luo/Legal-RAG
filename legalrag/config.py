from __future__ import annotations

from pathlib import Path 
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

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

    law_raw: str = str(DATA_DIR / "raw" / "minfadian_hetongbian.txt")
    law_jsonl: str = str(DATA_DIR / "processed" / "law.jsonl")
    law_graph_jsonl: str = str(DATA_DIR / "graph" / "law_graph.jsonl")
    legal_kg_jsonl: str = str(DATA_DIR / "graph" / "legal_kg.jsonl")


class LLMConfig(BaseModel):
    provider: str = "qwen-local"   # "qwen-local" | "openai"

    # 默认值 
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    qwen_model: str = "Qwen/Qwen2.5-3B-Instruct"  
    openai_model: str = "gpt-5-nano" # "gpt-4o-mini"

    # 可选 env 覆盖 
    qwen_model_env: str = "QWEN_MODEL"
    openai_model_env: str = "OPENAI_MODEL"

    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    max_context_tokens: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.5
    top_p: float = 0.9
    repetition_penalty: float = 1.12
    no_repeat_ngram_size: int = 0  # 0 表示不启用    

class RetrievalConfig(BaseModel):
    # -------- Corpus --------
    processed_file: str = "processed/law.jsonl"
    processed_dir: str = "data/processed"
    processed_glob: str = "*.jsonl"

    # -------- FAISS (Dense) --------
    faiss_index_file: str = "index/faiss.index"
    faiss_meta_file: str = "index/faiss_meta.jsonl"
    embedding_model: str = "BAAI/bge-base-zh-v1.5"  # BAAI/bge-m3

    hnsw_m: int = 64
    hnsw_ef_construction: int = 400
    hnsw_ef_search: int = 512  # 可在运行时动态调整

    # -------- BM25 (Sparse) --------
    bm25_index_file: str = "index/bm25.pkl"

    # ------------ Graph ------------
    enable_graph: bool = True
    graph_seed_k: int = 30                # 选多少 seed 去扩展 
    graph_walk_depths: Dict[str, int] = {
        "defined_by": 4,          
        "defines_term": 3,        
        "cite": 1,              
        "cited_by": 1,
        "prev": 2,             
        "next": 2,
        "default": 2
    }              
    graph_limit: int = 800            
    graph_weight: float = 0.2
    graph_rel_types: Optional[List[str]] = None

    # -------- Retrieval control --------
    top_k: int = 10
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    min_final_score: float = 0.2

    # -------- ColBERT (Late Interaction) --------
    enable_colbert: bool = True
    colbert_index_path: str = "index/colbert"
    colbert_meta_file: str = "index/colbert_meta.jsonl"
    colbert_weight: float = 0.35
    colbert_backend: str = "pylate"          # or "auto"
    colbert_index_name: str = "index"
    colbert_model_name: str =  "colbert-ir/colbertv2.0" # "lightonai/GTE-ModernColBERT-v1"  
    colbert_max_document_length: int = 300
    colbert_split_documents: bool = False
    colbert_overwrite: bool = True

    # -------- HyDE --------
    enable_hyde: bool = False

    # -------- Rerank --------
    enable_rerank: bool = True
    rerank_top_n: int = 30
    rrf_alpha: float = 0.5
    rerank_beta: float = 0.35
    rerank_ce_model: str = "BAAI/bge-reranker-base"

    # -------- Fusion --------
    fusion_method: str = "rrf_norm_blend"
    rrf_k: int = 60
    rrf_blend_alpha: float = 0.6


class PDFConfig(BaseModel):
    enable_ocr: bool = True
    ocr_lang: str = "chi_sim"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class RoutingConfig(BaseModel):
    enable_router: bool = True
    llm_based: bool = True


class AppConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
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
        r.colbert_index_path = abs_path(r.colbert_index_path)
        r.colbert_meta_file = abs_path(r.colbert_meta_file )

        Path(cfg.paths.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.index_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.eval_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.graph_dir).mkdir(parents=True, exist_ok=True)

        return cfg
