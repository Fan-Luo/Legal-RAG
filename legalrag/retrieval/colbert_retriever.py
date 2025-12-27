from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk


class ColBERTRetriever:
    """ColBERT channel with PyLate-first backend and RAGatouille fallback."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval

        self.enabled: bool = bool(getattr(rcfg, "enable_colbert", False))
        self.backend_pref: str = str(getattr(rcfg, "colbert_backend", "auto")).lower()

        self.index_path: Optional[str] = getattr(rcfg, "colbert_index_path", None)
        self.index_name: str = str(getattr(rcfg, "colbert_index_name", "index"))
        self.model_name: Optional[str] = getattr(rcfg, "colbert_model_name", None)

        self.meta_file: str = str(getattr(rcfg, "colbert_meta_file", "index/colbert_meta.jsonl"))

        self._id2chunk: Dict[str, LawChunk] = {}

        self._backend: str = "disabled"
        self._pylate_model = None
        self._pylate_index = None
        self._pylate_retriever = None
        self._rag_model = None  # ragatouille RAGPretrainedModel

        if not self.enabled:
            return

        if not self.index_path:
            raise RuntimeError("enable_colbert=True but cfg.retrieval.colbert_index_path is not set")

        self._load_meta()
        self._init_backend()

    def _load_meta(self) -> None:
        meta_path = Path(self.meta_file)
        if not meta_path.exists():
            raise RuntimeError(f"ColBERT meta file not found: {meta_path}")

        id2chunk: Dict[str, LawChunk] = {}
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj.get("id")
                if cid is None:
                    continue
                id2chunk[str(cid)] = LawChunk(**obj)

        if not id2chunk:
            raise RuntimeError(f"ColBERT meta file is empty: {meta_path}")

        self._id2chunk = id2chunk

    def _init_backend(self) -> None:
        errors: List[str] = []

        if self.backend_pref in ("auto", "pylate"):
            try:
                self._init_pylate()
                self._backend = "pylate"
                return
            except Exception as e:
                errors.append(f"[pylate] {type(e).__name__}: {e}")
                if self.backend_pref == "pylate":
                    raise RuntimeError(
                        "ColBERT backend set to 'pylate' but initialization failed.\n"
                        + "\n".join(errors)
                    ) from e

        if self.backend_pref in ("auto", "ragatouille"):
            try:
                self._init_ragatouille()
                self._backend = "ragatouille"
                return
            except Exception as e:
                errors.append(f"[ragatouille] {type(e).__name__}: {e}")
                if self.backend_pref == "ragatouille":
                    raise RuntimeError(
                        "ColBERT backend set to 'ragatouille' but initialization failed.\n"
                        + "\n".join(errors)
                    ) from e

        raise RuntimeError(
            "Failed to initialize any ColBERT backend.\n"
            "Tried (in order): pylate, ragatouille.\n\n"
            "Errors:\n  - " + "\n  - ".join(errors) + "\n\n"
            "Actionable fixes:\n"
            "  1) Preferred: pip install pylate (then rebuild ColBERT index with PyLate builder)\n"
            "  2) Fallback: keep ragatouille, but Stanford ColBERT may require a compatible "
            "torch/setuptools toolchain and is fragile on some runtimes (e.g., Py3.11).\n"
        )

    def _init_pylate(self) -> None:
        if not self.model_name:
            raise RuntimeError(
                "PyLate backend requires cfg.retrieval.colbert_model_name "
                "(e.g., 'lightonai/GTE-ModernColBERT-v1')."
            )

        try:
            from pylate import indexes, models, retrieve  
        except Exception as e:
            raise RuntimeError("PyLate is not installed or failed to import. Install with: pip install pylate") from e

        index_folder = str(self.index_path)

        self._pylate_model = models.ColBERT(model_name_or_path=self.model_name)
        self._pylate_index = indexes.PLAID(
            index_folder=index_folder,
            index_name=self.index_name,
            override=False,
            use_fast=False,  # IMPORTANT: disable FastPlaid (Rust) to avoid ABI mismatch
        )
        self._pylate_retriever = retrieve.ColBERT(index=self._pylate_index)

    def _init_ragatouille(self) -> None:
        try:
            from ragatouille import RAGPretrainedModel 
        except Exception as e:
            raise RuntimeError("RAGatouille failed to import. Install with: pip install 'ragatouille<0.0.10'") from e

        self._rag_model = RAGPretrainedModel.from_index(self.index_path)

    def search(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        if not self.enabled:
            return []

        k = int(top_k)
        if k <= 0:
            return []

        if self._backend == "pylate":
            return self._search_pylate(query, k)
        if self._backend == "ragatouille":
            return self._search_ragatouille(query, k)
        return []

    def _search_pylate(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        model = self._pylate_model
        retriever = self._pylate_retriever
        if model is None or retriever is None:
            return []

        queries_embeddings = model.encode([query], is_query=True, show_progress_bar=False)
        results = retriever.retrieve(queries_embeddings=queries_embeddings, k=top_k)

        hits: List[Tuple[LawChunk, float]] = []
        for item in (results[0] if results else []):
            cid = str(item.get("id"))
            chunk = self._id2chunk.get(cid)
            if chunk is None:
                continue
            hits.append((chunk, float(item.get("score", 0.0))))
        return hits

    def _search_ragatouille(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        rag = self._rag_model
        if rag is None:
            return []

        res = rag.search(query, k=top_k)
        hits: List[Tuple[LawChunk, float]] = []
        for r in res:
            cid = r.get("document_id") or r.get("doc_id") or r.get("id")
            if cid is None:
                continue
            chunk = self._id2chunk.get(str(cid))
            if chunk is None:
                continue
            hits.append((chunk, float(r.get("score", 0.0))))
        return hits
