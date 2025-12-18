# legalrag/retrieval/incremental_indexer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
from filelock import FileLock

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.retrieval.vector_store import VectorStore
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class IncrementalIndexer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.vs = VectorStore(cfg)

    def _load_jsonl_chunks(self, jsonl_path: Path) -> List[LawChunk]:
        chunks: List[LawChunk] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(LawChunk(**json.loads(line)))
        return chunks

    def add_jsonl(self, jsonl_path: str | Path) -> int:
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            logger.error(f"[index] jsonl not found: {jsonl_path}")
            raise FileNotFoundError(jsonl_path)

        logger.info(f"[index] start incremental add: {jsonl_path}")

        incoming = self._load_jsonl_chunks(jsonl_path)
        logger.info(f"[index] loaded chunks from jsonl: {len(incoming)}")
        if not incoming:
            logger.warning(f"[index] empty jsonl, skip: {jsonl_path}")
            return 0

        lock_path = Path(self.cfg.retrieval.faiss_index_file).with_suffix(".lock")
        with FileLock(str(lock_path)):
            # 1) load existing index + meta
            self.vs.load()

            exist_ids = {c.id for c in self.vs.chunks}
            new_chunks = [c for c in incoming if c.id not in exist_ids]
            added = len(new_chunks)

            logger.info(
                f"[index] dedup done: incoming={len(incoming)} exist={len(exist_ids)} added={added}"
            )
            if added == 0:
                return 0

            # 2) embed + add to in-memory index
            vecs = self.vs._embed([c.text for c in new_chunks]).astype("float32")
            self.vs.index.add(vecs)

            # 3) append meta first (avoid index/meta mismatch causing search idx->chunk crash)
            self.vs.meta_path.parent.mkdir(parents=True, exist_ok=True)
            with self.vs.meta_path.open("a", encoding="utf-8") as f:
                for c in new_chunks:
                    f.write(c.model_dump_json() + "\n")

            # keep in-memory chunks consistent as well
            self.vs.chunks.extend(new_chunks)

            # 4) persist faiss index
            self.vs.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.vs.index, str(self.vs.index_path))

        logger.info(f"[index] incremental add done: added={added} jsonl={jsonl_path}")
        return added
