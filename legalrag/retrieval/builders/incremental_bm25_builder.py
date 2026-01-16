from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List

import jieba
from rank_bm25 import BM25Okapi

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class IncrementalBM25Builder:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def _load_jsonl_chunks(self, jsonl_path: Path) -> List[LawChunk]:
        chunks: List[LawChunk] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(LawChunk(**json.loads(line)))
        return chunks

    def _load_existing(self) -> List[LawChunk]:
        bm25_path = Path(self.cfg.retrieval.bm25_index_file)
        if not bm25_path.exists():
            return []
        try:
            with bm25_path.open("rb") as f:
                payload = pickle.load(f)
            chunks = payload.get("chunks") or []
            return [LawChunk.model_validate(c) for c in chunks]
        except Exception:
            return []

    def add_jsonl(self, jsonl_path: str | Path) -> int:
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            logger.error("[bm25] jsonl not found: %s", jsonl_path)
            raise FileNotFoundError(jsonl_path)

        logger.info("[bm25] start incremental add: %s", jsonl_path)
        incoming = self._load_jsonl_chunks(jsonl_path)
        if not incoming:
            logger.warning("[bm25] empty jsonl, skip: %s", jsonl_path)
            return 0

        existing = self._load_existing()
        exist_ids = {c.id for c in existing}
        new_chunks = [c for c in incoming if c.id not in exist_ids]
        if not new_chunks:
            dup_ids = [c.id for c in incoming if c.id in exist_ids]
            sample = dup_ids[:5]
            logger.info(
                "[bm25] no new chunks to add: incoming=%d duplicate=%d sample_ids=%s",
                len(incoming),
                len(dup_ids),
                sample,
            )
            return 0

        all_chunks = existing + new_chunks
        corpus_tokens = [list(jieba.cut(c.text)) for c in all_chunks]
        bm25 = BM25Okapi(corpus_tokens)

        bm25_path = Path(self.cfg.retrieval.bm25_index_file)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"bm25": bm25, "chunks": [c.model_dump() for c in all_chunks]}
        tmp_path = bm25_path.with_suffix(".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, bm25_path)

        logger.info("[bm25] incremental add done: added=%d total=%d", len(new_chunks), len(all_chunks))
        return len(new_chunks)
