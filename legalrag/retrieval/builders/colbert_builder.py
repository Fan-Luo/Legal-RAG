from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from filelock import FileLock

from legalrag.schemas import LawChunk


def _chunk_to_dict(c: LawChunk) -> dict:
    if hasattr(c, "model_dump"):
        return c.model_dump()
    if hasattr(c, "dict"):
        return c.dict()
    return dict(c.__dict__)


def _write_meta(meta_file: str, chunks: Iterable[LawChunk]) -> None:
    p = Path(meta_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(_chunk_to_dict(c), ensure_ascii=False) + "\n")


def build_colbert_index(cfg: Any, chunks: List[LawChunk], override: bool = True) -> None:
    """Build ColBERT index using PyLate-first strategy.

    This builder:
      - builds the actual ColBERT/PLAID index
      - writes meta jsonl (id -> LawChunk) used by ColBERTRetriever
    """
    if not chunks:
        raise ValueError("build_colbert_index: chunks is empty")

    rcfg = cfg.retrieval
    backend_pref = str(getattr(rcfg, "colbert_backend", "auto")).lower()
    index_path: Optional[str] = getattr(rcfg, "colbert_index_path", None)
    index_name: str = str(getattr(rcfg, "colbert_index_name", "index"))
    model_name: Optional[str] = getattr(rcfg, "colbert_model_name", None)
    meta_file: str = str(getattr(rcfg, "colbert_meta_file", "index/colbert_meta.jsonl"))

    if not index_path:
        raise RuntimeError("cfg.retrieval.colbert_index_path is required to build ColBERT index")

    lock = FileLock(str(Path(index_path).with_suffix(".lock")))
    with lock:
        errors: List[str] = []

        if backend_pref in ("auto", "pylate"):
            try:
                _build_pylate(index_path=index_path, index_name=index_name, model_name=model_name, chunks=chunks, override=override)
                _write_meta(meta_file, chunks)
                return
            except Exception as e:
                errors.append(f"[pylate] {type(e).__name__}: {e}")
                if backend_pref == "pylate":
                    raise

        if backend_pref in ("auto", "ragatouille"):
            try:
                _build_ragatouille(index_path=index_path, index_name=index_name, model_name=model_name, chunks=chunks, override=override)
                _write_meta(meta_file, chunks)
                return
            except Exception as e:
                errors.append(f"[ragatouille] {type(e).__name__}: {e}")
                if backend_pref == "ragatouille":
                    raise

        raise RuntimeError(
            "Failed to build ColBERT index with any backend.\n"
            "Tried (in order): pylate, ragatouille.\n\n"
            "Errors:\n  - " + "\n  - ".join(errors) + "\n\n"
            "Actionable fixes:\n"
            "  1) Preferred: pip install pylate and set cfg.retrieval.colbert_model_name\n"
            "  2) Fallback: ragatouille requires Stanford ColBERT toolchain; may be fragile on Py3.11\n"
        )


def _build_pylate(*, index_path: str, index_name: str, model_name: Optional[str], chunks: List[LawChunk], override: bool) -> None:
    if not model_name:
        raise RuntimeError("PyLate builder requires cfg.retrieval.colbert_model_name (model_name_or_path)")

    try:
        from pylate import indexes, models  
    except Exception as e:
        raise RuntimeError("PyLate is not installed or failed to import. Install with: pip install pylate") from e

    docs = [c.text for c in chunks]
    ids = [str(c.id) for c in chunks]

    model = models.ColBERT(model_name_or_path=model_name)
    index = indexes.PLAID(
        index_folder=str(index_path),
        index_name=str(index_name),
        override=bool(override),
        use_fast=False,   
    )

    emb = model.encode(docs, is_query=False, show_progress_bar=True)
    index.add_documents(documents_ids=ids, documents_embeddings=emb)


def _build_ragatouille(*, index_path: str, index_name: str, model_name: Optional[str], chunks: List[LawChunk], override: bool) -> None:
    """ ragatouille builder fallback. """
    try:
        from ragatouille import RAGPretrainedModel  
    except Exception as e:
        raise RuntimeError("RAGatouille failed to import. Install with: pip install 'ragatouille<0.0.10'") from e

    ckpt = model_name or "colbert-ir/colbertv2.0"
    rag = RAGPretrainedModel.from_pretrained(ckpt)

    docs = [c.text for c in chunks]
    ids = [str(c.id) for c in chunks]

    # Try common signatures across ragatouille versions.
    try:
        rag.index(collection=docs, document_ids=ids, index_name=index_name, index_root=index_path, overwrite=bool(override))
        return
    except TypeError:
        pass

    try:
        rag.index(collection=docs, document_ids=ids, index_name=index_name, index_path=index_path, overwrite=bool(override))
        return
    except TypeError:
        pass

    raise RuntimeError(
        "Unable to call ragatouille.index() with known signatures for this version. "
        "Switch to PyLate backend (recommended) or adapt this fallback to your ragatouille API."
    )
