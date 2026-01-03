from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from filelock import FileLock

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk


def _chunk_to_dict(c: LawChunk) -> dict: 
    if hasattr(c, "model_dump"):
        return c.model_dump()
    if hasattr(c, "dict"):
        return c.dict()  # type: ignore[attr-defined]
    return {
        "id": getattr(c, "id", None),
        "article_id": getattr(c, "article_id", None),
        "article_no": getattr(c, "article_no", None),
        "law_name": getattr(c, "law_name", None),
        "chapter": getattr(c, "chapter", None),
        "section": getattr(c, "section", None),
        "title": getattr(c, "title", None),
        "text": getattr(c, "text", None),
        "source": getattr(c, "source", None),
        "start_char": getattr(c, "start_char", None),
        "end_char": getattr(c, "end_char", None),
    }


def _write_meta(meta_file: str, chunks: List[LawChunk]) -> None:
    """Persist chunk metadata for retriever lookup: chunk.id -> LawChunk."""
    p = Path(meta_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            obj = _chunk_to_dict(c)
            # Store by chunk.id; retriever maps back from doc_id -> chunk
            obj["_doc_id"] = str(getattr(c, "id", ""))
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _build_pylate_voyager(
    *,
    index_path: str,
    index_name: str,
    model_name: str,
    chunks: List[LawChunk],
    override: bool,
) -> None:
    """
    Build ColBERT (late-interaction) index with PyLate Voyager backend.
    This builder encodes all docs with models.ColBERT and writes embeddings into the Voyager index.
    """
    try:
        from pylate import indexes, models
    except Exception as e:  
        raise RuntimeError("PyLate is not installed or failed to import. Install with: pip install pylate") from e

    if not hasattr(indexes, "Voyager"):
        raise RuntimeError(
            "Your installed pylate does not expose indexes.Voyager. "
            "Please upgrade pylate to a version that includes the Voyager backend."
        )

    docs: List[str] = [(c.text or "").strip() for c in chunks]
    ids: List[str] = [str(getattr(c, "id", "")) for c in chunks]

    # Filter empty docs/ids defensively
    filtered_docs: List[str] = []
    filtered_ids: List[str] = []
    for d, cid in zip(docs, ids):
        if cid and d:
            filtered_docs.append(d)
            filtered_ids.append(cid)

    if not filtered_docs:
        raise RuntimeError("No non-empty documents found to build ColBERT index.")

    model = models.ColBERT(model_name_or_path=model_name)

    idx = indexes.Voyager(
        index_folder=str(index_path),
        index_name=str(index_name),
        override=bool(override),
    )

    # Encode documents (ColBERT doc embeddings)
    # PyLate API: model.encode(texts, is_query=..., show_progress_bar=...)
    doc_emb = model.encode(filtered_docs, is_query=False, show_progress_bar=True)

 
    ingested = False

    if hasattr(idx, "add_documents"):
        try:
            idx.add_documents(documents_ids=filtered_ids, documents_embeddings=doc_emb)
            ingested = True
        except TypeError: 
            try:
                idx.add_documents(filtered_ids, doc_emb)
                ingested = True
            except Exception:
                pass

        if not ingested:
            try: 
                idx.add_documents(documents=filtered_docs, documents_ids=filtered_ids)
                ingested = True
            except Exception:
                pass

    if not ingested and hasattr(idx, "add"):
        try:
            idx.add(filtered_ids, doc_emb)
            ingested = True
        except Exception:
            pass

    if not ingested and hasattr(idx, "index"):
        try:
            idx.index(documents=filtered_docs, document_ids=filtered_ids, embeddings=doc_emb)
            ingested = True
        except Exception:
            pass

    if not ingested:
        raise RuntimeError(
            "Unable to ingest documents into pylate.indexes.Voyager with known method signatures. "
            "Inspect your pylate version's Voyager API and adapt _build_pylate_voyager accordingly."
        )

    # Persist index if required by backend
    if hasattr(idx, "save"):
        try:
            idx.save()
        except Exception:
            pass
    if hasattr(idx, "persist"):
        try:
            idx.persist()
        except Exception:
            pass


def build_colbert_index(cfg: AppConfig, chunks: List[LawChunk], override: bool = False) -> None:

    rcfg = cfg.retrieval
    enabled = bool(getattr(rcfg, "enable_colbert", False))
    if not enabled:
        return

    index_path = str(getattr(rcfg, "colbert_index_path", "")).strip()
    if not index_path:
        raise RuntimeError("enable_colbert=True but cfg.retrieval.colbert_index_path is not set")

    index_name = str(getattr(rcfg, "colbert_index_name", "index"))
    model_name: Optional[str] = getattr(rcfg, "colbert_model_name", None)
    if not model_name:
        raise RuntimeError(
            "enable_colbert=True but cfg.retrieval.colbert_model_name is not set "
            "(e.g., 'colbert-ir/colbertv2.0')."
        )

    meta_file = str(getattr(rcfg, "colbert_meta_file", "index/colbert_meta.jsonl"))

    lock = FileLock(str(Path(index_path).with_suffix(".lock")))
    with lock:
        Path(index_path).mkdir(parents=True, exist_ok=True)

        _build_pylate_voyager(
            index_path=index_path,
            index_name=index_name,
            model_name=model_name,
            chunks=chunks,
            override=override,
        )

        _write_meta(meta_file, chunks)
