from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from ragatouille import RAGPretrainedModel  


class ColBERTRetriever:
    """
    ColBERT / late-interaction retriever backed by a prebuilt RAGatouille index.

    Required artifacts (built by build_colbert_index.py):
      - cfg.retrieval.colbert_index_path: directory containing the ColBERT index
      - cfg.retrieval.colbert_meta_file: jsonl mapping chunk.id -> LawChunk (full metadata)
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval

        self.enabled = bool(getattr(rcfg, "enable_colbert", False))
        self.index_path = getattr(rcfg, "colbert_index_path", None)
        self.meta_file = getattr(rcfg, "colbert_meta_file", "index/colbert_meta.jsonl")

        self._id2chunk: Dict[str, LawChunk] = {}
        self._model = None

        if not self.enabled:
            return
        if RAGPretrainedModel is None:
            raise RuntimeError("ColBERTRetriever requires `ragatouille`. Install with: pip install ragatouille")
        if not self.index_path:
            raise RuntimeError("cfg.retrieval.colbert_index_path is required when enable_colbert=True")

        self._model = RAGPretrainedModel.from_index(self.index_path)
        self._load_meta()

    def _load_meta(self) -> None:
        meta_path = Path(self.meta_file)
        if not meta_path.exists():
            raise RuntimeError(f"ColBERTRetriever meta file not found: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                chunk = LawChunk(**obj)
                self._id2chunk[str(chunk.id)] = chunk

        if not self._id2chunk:
            raise RuntimeError(f"ColBERTRetriever meta file is empty: {meta_path}")

    def search(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        """
        Returns list of (LawChunk, score). Score is ColBERT similarity (higher is better).

        RAGatouille result keys vary by version; we handle common patterns:
          - document_id / doc_id / id
          - score
        """
        if not self.enabled or self._model is None:
            return []

        res = self._model.search(query, k=int(top_k))
        hits: List[Tuple[LawChunk, float]] = []

        for r in res:
            cid = r.get("document_id") or r.get("doc_id") or r.get("id")
            if cid is None:
                continue
            cid = str(cid)

            chunk = self._id2chunk.get(cid)
            if chunk is None:
                continue

            score = float(r.get("score", 0.0))
            hits.append((chunk, score))

        return hits
