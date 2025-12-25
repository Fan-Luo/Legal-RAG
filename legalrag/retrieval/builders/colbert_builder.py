from __future__ import annotations

from pathlib import Path
from typing import List

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def build_colbert_index(cfg: AppConfig, chunks: List[LawChunk]) -> None:
    """
    Build a ColBERTv2 index using RAGatouille.

    Requires:
      pip install ragatouille

    Writes:
      - cfg.retrieval.colbert_index_path (directory)
      - cfg.retrieval.colbert_meta_file (jsonl mapping chunk.id -> LawChunk)
    """
    try:
        from ragatouille import RAGPretrainedModel  # type: ignore
    except Exception as e:
        raise RuntimeError("ColBERT indexing requires `ragatouille`. Install with: pip install ragatouille") from e

    rcfg = cfg.retrieval
    index_path = Path(getattr(rcfg, "colbert_index_path", "index/colbert"))
    meta_file = Path(getattr(rcfg, "colbert_meta_file", "index/colbert_meta.jsonl"))
    model_name = getattr(rcfg, "colbert_model_name", "colbert-ir/colbertv2.0")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    collection = [c.text for c in chunks]
    doc_ids = [str(c.id) for c in chunks]

    max_doc_len = int(getattr(rcfg, "colbert_max_document_length", 300))
    split_docs = bool(getattr(rcfg, "colbert_split_documents", False))
    overwrite = bool(getattr(rcfg, "colbert_overwrite", True))

    logger.info("[ColBERT] model=%s", model_name)
    colbert = RAGPretrainedModel.from_pretrained(model_name)

    logger.info(
        "[ColBERT] building index at %s (N=%d, max_document_length=%d, split=%s, overwrite=%s)",
        index_path, len(collection), max_doc_len, split_docs, overwrite
    )

    try:
        colbert.index(
            collection=collection,
            document_ids=doc_ids,
            index_name=index_path.name,
            index_path=str(index_path),
            max_document_length=max_doc_len,
            split_documents=split_docs,
            overwrite=overwrite,
        )
    except TypeError:
        colbert.index(
            collection,
            index_name=index_path.name,
            max_document_length=max_doc_len,
            split_documents=split_docs,
            overwrite=overwrite,
        )
        logger.warning(
            "[ColBERT] ragatouille may not support `document_ids`/`index_path` in this version. "
            "Upgrade recommended for stable id mapping and deterministic index path."
        )

    with meta_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    logger.info("[ColBERT] meta written: %s", meta_file)
    logger.info("[ColBERT] index built: %s", index_path)
