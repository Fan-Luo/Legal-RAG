from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from filelock import FileLock
from legalrag.retrieval.builders.incremental_dense_builder import IncrementalDenseBuilder
from legalrag.retrieval.builders.bm25_builder import build_bm25_index
from legalrag.retrieval.builders.colbert_builder import build_colbert_index
from legalrag.retrieval.builders.incremental_bm25_builder import IncrementalBM25Builder
from legalrag.retrieval.builders.graph_builder import GraphBuilder
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class IngestOrchestrator:
    cfg: Any
    status: Dict[str, Dict[str, Any]]

    def _load_corpus(self):
        corpus = Path(self.cfg.paths.law_jsonl)
        chunks = []
        with corpus.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(LawChunk.model_validate_json(line))
        logger.info("[Ingest] loaded corpus chunks=%d from %s", len(chunks), corpus)
        return chunks

    def faiss_job(self, jsonl_path: str, doc_id: str):
        try:
            self.status[doc_id]["faiss"] = "running"
            logger.info("[Ingest][FAISS] doc_id=%s jsonl=%s", doc_id, jsonl_path)
            added = IncrementalDenseBuilder(self.cfg).add_jsonl(jsonl_path)
            self.status[doc_id].update({"faiss": "done", "added": added})
        except Exception as e:
            self.status[doc_id].update({"faiss": "failed", "error": str(e)})

    def bm25_job(self, jsonl_path: str, doc_id: str):
        try:
            self.status[doc_id]["bm25"] = "running"
            logger.info("[Ingest][BM25] doc_id=%s jsonl=%s", doc_id, jsonl_path)
            with FileLock(self.cfg.retrieval.bm25_index_file + ".lock"):
                added = IncrementalBM25Builder(self.cfg).add_jsonl(jsonl_path)
            self.status[doc_id]["bm25"] = "done"
        except Exception as e:
            self.status[doc_id].update({"bm25": "failed", "error": str(e)})

    def colbert_job(self, doc_id: str):
        try:
            self.status[doc_id]["colbert"] = "running"
            if not getattr(self.cfg.retrieval, "ingest_rebuild_colbert", True):
                self.status[doc_id]["colbert"] = "skipped"
                return
            logger.info("[Ingest][ColBERT] doc_id=%s rebuild=1", doc_id)
            chunks = self._load_corpus()
            build_colbert_index(self.cfg, chunks, override=True)
            self.status[doc_id]["colbert"] = "done"
        except Exception as e:
            self.status[doc_id].update({"colbert": "failed", "error": str(e)})

    def graph_job(self, doc_id: str):
        try:
            self.status[doc_id]["graph"] = "running"
            if not getattr(self.cfg.retrieval, "ingest_rebuild_graph", True):
                self.status[doc_id]["graph"] = "skipped"
                return
            logger.info("[Ingest][Graph] doc_id=%s rebuild=1", doc_id)
            chunks = self._load_corpus()
            GraphBuilder(self.cfg).build_from_chunks(chunks)
            self.status[doc_id]["graph"] = "done"
        except Exception as e:
            self.status[doc_id].update({"graph": "failed", "error": str(e)})
