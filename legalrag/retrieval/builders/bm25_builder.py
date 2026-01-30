from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import jieba
import re
from rank_bm25 import BM25Okapi

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize_en(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def build_bm25_index(cfg: AppConfig, chunks: List[LawChunk]) -> None:
    """
    Build BM25 index and persist to cfg.retrieval.bm25_index_file.

    Output format (stable):
      {"bm25": BM25Okapi, "chunks": [<LawChunk dict>, ...]}

    Notes:
      - We store chunks as dicts (not pickled LawChunk objects) to reduce pickle brittleness.
      - Tokenization uses jieba (aligned with your current implementation).
    """
    rcfg = cfg.retrieval
    bm25_path = Path(rcfg.bm25_index_file)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[BM25] building (docs=%d) -> %s", len(chunks), bm25_path)

    lang = (getattr(chunks[0], "lang", None) or "zh").strip().lower() if chunks else "zh"
    if lang == "en":
        corpus_tokens = [_tokenize_en(c.text) for c in chunks]
    else:
        corpus_tokens = [list(jieba.cut(c.text)) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "chunks": [c.model_dump() for c in chunks],
    }
    with bm25_path.open("wb") as f:
        pickle.dump(payload, f)

    logger.info("[BM25] saved -> %s", bm25_path)
