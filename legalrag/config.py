from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
 
class AppConfig(BaseModel):
    paths: PathsConfig = PathsConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig() 

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

        return cfg
