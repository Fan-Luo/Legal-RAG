from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from legalrag.config import AppConfig
from legalrag.models import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.retrieval
        self.index_path = Path(rcfg.faiss_index_file)
        self.meta_path = Path(rcfg.faiss_meta_file)
        self.model = SentenceTransformer(rcfg.embedding_model)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[LawChunk] = []

    def load(self):
        if self.index is not None and self.chunks:
            return
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("FAISS 索引或元数据不存在，请先运行 build_index.")
        self.index = faiss.read_index(str(self.index_path))
        self.chunks = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(LawChunk(**obj))
        logger.info(f"[FAISS] Loaded {len(self.chunks)} chunks")

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        return emb.astype("float32")

    def search(self, query: str, top_k: int) -> List[Tuple[LawChunk, float]]:
        self.load()
        q_vec = self._embed([query])
        scores, idxs = self.index.search(q_vec, top_k)
        hits: List[Tuple[LawChunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            hits.append((self.chunks[idx], float(score)))
        return hits
