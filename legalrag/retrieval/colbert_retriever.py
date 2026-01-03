from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk


class ColBERTRetriever:
    """
    ColBERT channel retriever using PyLate Voyager backend only.

    Contract:
      - search(query, top_k) -> List[(LawChunk, score)]
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval

        self.enabled: bool = bool(getattr(rcfg, "enable_colbert", False))
        self.index_path: Path = Path(str(getattr(rcfg, "colbert_index_path", "index/colbert")))
        self.index_name: str = str(getattr(rcfg, "colbert_index_name", "index"))

        self.model_name: Optional[str] = getattr(rcfg, "colbert_model_name", None)
        self.meta_file: str = str(getattr(rcfg, "colbert_meta_file", "index/colbert_meta.jsonl"))

        self._id2chunk: Dict[str, LawChunk] = {}

        self._pylate_model = None
        self._pylate_index = None
        self._pylate_retriever = None

        if not self.enabled:
            return

        if not self.model_name:
            raise RuntimeError(
                "enable_colbert=True but cfg.retrieval.colbert_model_name is not set "
                "(e.g., 'colbert-ir/colbertv2.0')."
            )

        self._load_meta()
        self._init_pylate_voyager()

    def _load_meta(self) -> None:
        meta_path = Path(self.meta_file)
        if not meta_path.exists():
            raise RuntimeError(f"ColBERT meta file not found: {meta_path}")

        id2chunk: Dict[str, LawChunk] = {}
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = str(obj.get("_doc_id") or obj.get("id") or "")
                if not doc_id:
                    continue
                # Remove helper field if present
                obj.pop("_doc_id", None)
                try:
                    # Pydantic v2
                    chunk = LawChunk(**obj)
                except Exception:
                    # best-effort fallback: keep minimum fields
                    chunk = LawChunk(
                        id=obj.get("id", doc_id),
                        article_id=obj.get("article_id", ""),
                        article_no=obj.get("article_no", ""),
                        law_name=obj.get("law_name", ""),
                        chapter=obj.get("chapter", ""),
                        section=obj.get("section", ""),
                        title=obj.get("title", None),
                        text=obj.get("text", ""),
                        source=obj.get("source", ""),
                        start_char=obj.get("start_char", None),
                        end_char=obj.get("end_char", None),
                    )
                id2chunk[doc_id] = chunk

        self._id2chunk = id2chunk

    def _init_pylate_voyager(self) -> None:
        try:
            from pylate import indexes, models, retrieve
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyLate is not installed or failed to import. Install with: pip install pylate") from e

        if not hasattr(indexes, "Voyager"):
            raise RuntimeError(
                "Your installed pylate does not expose indexes.Voyager. "
                "Please upgrade pylate to a version that includes the Voyager backend."
            )

        self._pylate_model = models.ColBERT(model_name_or_path=self.model_name)
        self._pylate_index = indexes.Voyager(
            index_folder=str(self.index_path),
            index_name=str(self.index_name),
            override=False,
        )
        self._pylate_retriever = retrieve.ColBERT(index=self._pylate_index)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[LawChunk, float]]:
        if not self.enabled:
            return []
        if not query:
            return []
        return self._search_pylate(query, top_k)

    def _search_pylate(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        model = self._pylate_model
        retriever = self._pylate_retriever

        if model is None or retriever is None:
            return []

        q_emb = model.encode([query], is_query=True, show_progress_bar=False)
        results = retriever.retrieve(queries_embeddings=q_emb, k=int(top_k))

        # results often looks like: [ [ {"id": "...", "score": ...}, ... ] ]
        # but can vary by version; support common shapes.
        batch = results[0] if isinstance(results, list) and results else results

        hits: List[Tuple[LawChunk, float]] = []

        if batch is None:
            return hits

        # Case 1: list of dicts
        if isinstance(batch, list):
            for item in batch:
                if isinstance(item, dict):
                    cid = str(item.get("id") or item.get("document_id") or item.get("doc_id") or "")
                    score = float(item.get("score", item.get("similarity", 0.0)) or 0.0)
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    cid = str(item[0])
                    score = float(item[1])
                else:
                    continue

                chunk = self._id2chunk.get(cid)
                if chunk is None:
                    continue
                hits.append((chunk, score))
            return hits

        # Case 2: dict with "documents"/"scores"
        if isinstance(batch, dict):
            ids = batch.get("ids") or batch.get("documents_ids") or batch.get("document_ids") or []
            scores = batch.get("scores") or batch.get("similarities") or []
            for cid, score in zip(ids, scores):
                cid = str(cid)
                chunk = self._id2chunk.get(cid)
                if chunk is None:
                    continue
                hits.append((chunk, float(score)))
            return hits

        return hits
