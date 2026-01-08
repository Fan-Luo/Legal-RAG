from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk


def _ensure_colbert_importable() -> None:
    try:
        import colbert    
    except Exception as e:
        raise RuntimeError(
            "ColBERT is not importable. Install the official Stanford ColBERT package "
            "(e.g., `pip install colbert-ai` or `conda install -c conda-forge colbert-ai`), "
            "or clone https://github.com/stanford-futuredata/ColBERT and add it to PYTHONPATH."
        ) from e


def _dict_to_chunk(d: dict) -> LawChunk: 
    if hasattr(LawChunk, "model_validate"):
        return LawChunk.model_validate(d)
    return LawChunk.parse_obj(d)  # type: ignore[attr-defined]


class ColBERTRetriever:
    """
    ColBERT channel retriever using the official Stanford ColBERT Searcher.

    Contract:
      - search(query, top_k) -> List[(LawChunk, score)]
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval

        self.enabled: bool = bool(getattr(rcfg, "enable_colbert", False))
        self.index_path: Path = Path(str(getattr(rcfg, "colbert_index_path")))
        self.index_name: str = str(getattr(rcfg, "colbert_index_name"))
        self.model_name: Optional[str] = getattr(rcfg, "colbert_model_name", "colbert-ir/colbertv2.0")   
        self.meta_file: Path = Path(str(getattr(rcfg, "colbert_meta_file", "index/colbert/colbert_meta.jsonl")))
        self.experiment: str = str(getattr(rcfg, "colbert_experiment"))
        self.nranks: int = int(getattr(rcfg, "colbert_nranks", 1))

        self._pid2chunk: Dict[int, LawChunk] = {}
        self._collection: List[str] = []
        self._searcher = None

        if not self.enabled:
            return

        _ensure_colbert_importable()
        self._load_meta_and_collection()
        self._init_searcher()

    def _load_meta_and_collection(self) -> None:
        if not self.meta_file.exists():
            raise RuntimeError(
                f"ColBERT meta file not found: {self.meta_file}. "
                "Run build_colbert_index() first."
            )

        pid2chunk: Dict[int, LawChunk] = {}
        max_pid = -1

        with self.meta_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = int(rec["pid"])
                chunk = _dict_to_chunk(rec["chunk"])
                pid2chunk[pid] = chunk
                max_pid = max(max_pid, pid)

        if max_pid < 0:
            raise RuntimeError(f"ColBERT meta file is empty: {self.meta_file}")

        # Reconstruct the collection in pid order (ColBERT passage ids refer to positions)
        collection: List[str] = [""] * (max_pid + 1)
        for pid, chunk in pid2chunk.items():
            collection[pid] = (getattr(chunk, "text", "") or "").strip()

        self._pid2chunk = pid2chunk
        self._collection = collection

    def _init_searcher(self) -> None:
        from colbert import Searcher
        from colbert.infra import Run, RunConfig

        with Run().context(RunConfig(root=str(self.index_path), nranks=self.nranks, experiment=self.experiment)):
            self._searcher = Searcher(index=self.index_name, collection=self._collection, checkpoint=self.model_name)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LawChunk, float]]:
        if not self.enabled:
            return []
        if not self._searcher:
            raise RuntimeError("ColBERT Searcher is not initialized.")

        query = (query or "").strip()
        if not query:
            return []

        results = self._searcher.search(query, k=top_k)
        pids, ranks, scores = results  # pids are ints (passage ids)

        out: List[Tuple[LawChunk, float]] = []
        for pid, score in zip(pids, scores):
            try:
                chunk = self._pid2chunk[int(pid)]
                out.append((chunk, float(score)))
            except Exception:
                continue
        return out
