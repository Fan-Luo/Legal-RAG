from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from filelock import FileLock

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk


def _chunk_to_dict(c: LawChunk) -> dict:
    """Serialize LawChunk in a pydantic-friendly way."""
    if hasattr(c, "model_dump"):
        return c.model_dump()
    if hasattr(c, "dict"):
        return c.dict()
    # Best-effort fallback
    return dict(c.__dict__)


def _ensure_colbert_importable() -> None:
    """
    Supported setups:
      1) pip/conda install of 'colbert-ai' (module name: 'colbert')
      2) local clone at ./ColBERT added to sys.path by the caller/notebook
    """
    try:
        import colbert  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "ColBERT is not importable. Install the official Stanford ColBERT package "
            "(e.g., `pip install colbert-ai` or `conda install -c conda-forge colbert-ai`), "
            "or clone https://github.com/stanford-futuredata/ColBERT and add it to PYTHONPATH."
        ) from e


def _write_meta(meta_file: Path, chunks: List[LawChunk]) -> None:
    """
    Write a JSONL mapping from ColBERT passage-id (pid) to the original LawChunk.

    We use pid == row index in the collection list passed to Indexer.index().
    """
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with meta_file.open("w", encoding="utf-8") as f:
        for pid, c in enumerate(chunks):
            rec = {
                "pid": pid,
                "chunk": _chunk_to_dict(c),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_colbert_index(
    cfg: AppConfig,
    chunks: List[LawChunk],
    override: bool = False
) -> Path:
    """
    Build an official ColBERTv2 (PLAID) index using Stanford ColBERT.

    Parameters
    ----------
    cfg:
        AppConfig. Uses:
          - cfg.retrieval.enable_colbert
          - cfg.retrieval.colbert_model_name
          - cfg.retrieval.colbert_index_path
          - cfg.retrieval.colbert_index_name
          - cfg.retrieval.colbert_meta_file
    chunks:
        List[LawChunk] where chunk.text is the passage string.
    override:
        If True, overwrite any existing index of the same name.
    nbits/doc_maxlen/kmeans_niters:
        Standard PLAID indexing parameters (see ColBERT docs/notebooks).
    nranks:
        Number of GPUs to use (1 for single GPU; 0 for CPU-only where supported).
    experiment:
        ColBERT "experiment" namespace. This becomes a folder under ColBERT's root.

    Returns
    -------
    Path:
        The absolute path to the built ColBERT index folder.
    """
    rcfg = cfg.retrieval
    enabled = bool(getattr(rcfg, "enable_colbert", False))
    if not enabled:
        raise RuntimeError("ColBERT is disabled: set cfg.retrieval.enable_colbert=True")

    model_name: Optional[str] = getattr(rcfg, "colbert_model_name", "colbert-ir/colbertv2.0") 
    index_path: Path = Path(str(getattr(rcfg, "colbert_index_path")))
    index_name: str = str(getattr(rcfg, "colbert_index_name"))
    meta_file: Path = Path(str(getattr(rcfg, "colbert_meta_file", "index/colbert/colbert_meta.jsonl")))
    nbits: int = int(getattr(rcfg, "colbert_nbits", 4))
    doc_maxlen: int = int(getattr(rcfg, "colbert_doc_maxlen", 220))
    kmeans_niters: int = int(getattr(rcfg, "colbert_kmeans_niters", 10))
    nranks: int = int(getattr(rcfg, "colbert_nranks", 1))
    experiment: str = str(getattr(rcfg, "colbert_experiment"))

    _ensure_colbert_importable()

    # Defensive: normalize docs
    docs: List[str] = [(getattr(c, "text", "") or "").strip() for c in chunks]
    if not any(docs):
        raise RuntimeError("All chunks are empty; cannot build ColBERT index.")

    # ColBERT writes under ColBERTConfig(root=...) and RunConfig(experiment=...)
    lock = FileLock(str(index_path / ".colbert_build.lock"))
    with lock:
        index_path.mkdir(parents=True, exist_ok=True)
        print('colbert index_path: ', str(index_path))

        # Meta is used by our system to map pid -> LawChunk for evidence display.
        _write_meta(meta_file, chunks)

        # Build index
        from colbert import Indexer
        from colbert.infra import Run, RunConfig, ColBERTConfig

        with Run().context(RunConfig(nranks=nranks, experiment=experiment, root=str(index_path))):
            config = ColBERTConfig(
                root=str(index_path),
                doc_maxlen=doc_maxlen,
                nbits=nbits,
                kmeans_niters=kmeans_niters,
                checkpoint=model_name
            )
            indexer = Indexer(checkpoint=model_name, config=config)
            indexer.index(name=index_name, collection=docs, overwrite=override)
 
            abs_index = Path(indexer.get_index()) 

    return abs_index
