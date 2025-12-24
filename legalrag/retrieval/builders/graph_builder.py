from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GraphBuilder:
    """
    Final-shape GraphBuilder.

    This builder is used by:
      - background GraphBuildJob (server ingest)
      - scripts/build_graph.py thin wrapper

    Each node connects to previous/next as neighbors (prev/next).
    """
    cfg: AppConfig

    def build_from_chunks(self, chunks: List[LawChunk]) -> Path:
        out_graph = Path(self.cfg.paths.law_graph_jsonl)
        out_graph.parent.mkdir(parents=True, exist_ok=True)
        out_graph.write_text("", encoding="utf-8")

        for i, c in enumerate(chunks):
            neighbors = []
            if i > 0:
                neighbors.append({"article_id": chunks[i - 1].article_id, "relation": "prev"})
            if i + 1 < len(chunks):
                neighbors.append({"article_id": chunks[i + 1].article_id, "relation": "next"})

            node = {
                "article_id": c.article_id,
                "article_no": getattr(c, "article_no", None),
                "law_name": getattr(c, "law_name", None),
                "chapter": getattr(c, "chapter", None),
                "section": getattr(c, "section", None),
                "neighbors": neighbors,
                "meta": {},
            }
            with out_graph.open("a", encoding="utf-8") as f:
                f.write(json.dumps(node, ensure_ascii=False) + "\n")

        logger.info("[GRAPH] built %d nodes -> %s", len(chunks), out_graph)
        return out_graph

    def build_from_corpus(self) -> Path:
        corpus = Path(self.cfg.paths.contract_law_jsonl)
        if not corpus.exists():
            raise FileNotFoundError(f"{corpus} not found; run preprocess first.")
        chunks = [json.loads(l) for l in corpus.open("r", encoding="utf-8") if l.strip()]
        # convert to LawChunk for consistent fields
        law_chunks: List[LawChunk] = []
        for obj in chunks:
            law_chunks.append(LawChunk.model_validate(obj) if hasattr(LawChunk, "model_validate") else LawChunk(**obj))
        return self.build_from_chunks(law_chunks)
