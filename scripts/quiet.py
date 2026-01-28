"""Quiet-mode utilities for notebooks and demos.

Goals:
- Suppress noisy logs (INFO/DEBUG), progress bars, and common framework warnings.
- Keep meaningful outputs (tables/figures/explicit prints) visible.
- Provide an opt-in context manager to silence extremely noisy calls.

Usage:
    from scripts.quiet import install_quiet
    install_quiet()

For a particularly noisy block:
    from scripts.quiet import suppress_output
    with suppress_output():
        do_noisy_work()
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Iterator, Optional


def install_quiet(
    *,
    python_warnings: bool = True,
    logging_level: int = logging.ERROR,
    silence_tqdm: bool = True,
    silence_hf_progress: bool = True,
    silence_tensorflow: bool = True,
    message_only_loggers: Optional[list[str]] = ['legalrag.retrieval.graph_store'],
    silence_colbert_stdout: bool = True,
) -> None:
    """Install a set of conservative defaults to reduce notebook noise.

    Parameters
    ----------
    python_warnings:
        If True, suppresses common warning categories that frequently clutter demo notebooks.
    logging_level:
        Root logging level to set (default ERROR). This suppresses INFO/DEBUG logs such as
        '2025-.. - __main__ - INFO - ...'.
    silence_tqdm:
        If True, tries to disable tqdm progress bars via environment variables.
    silence_hf_progress:
        If True, disables Hugging Face advisory warnings and some progress output.
    silence_tensorflow:
        If True, reduces TensorFlow / XLA / absl logging chatter where possible.
    """
    if python_warnings:
        # The jieba 'invalid escape sequence' lines come from SyntaxWarning during import.
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        # Reduce general deprecation/runtime warnings that are rarely useful in demos.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, message=r".*progress bar.*")
        # add more filters here 

    # Root logger: suppress INFO/DEBUG timestamped lines.
    logging.basicConfig(level=logging_level)
    logging.getLogger().setLevel(logging_level)


    for name in [
        "legalrag",
        "legalrag.retrieval",
        "legalrag.retrieval.graph_store",
        "legalrag.retrieval.hybrid_retriever",
        "legalrag.retrieval.bm25_retriever",
        "legalrag.pipeline.rag_pipeline",
        "legalrag.pipeline",
        "legalrag.api",
        "jieba",
        "transformers",
        "sentence_transformers",
        "faiss",
        "tensorflow",
        "torch",
        "urllib3",
    ]:
        logging.getLogger(name).setLevel(logging_level)


    for name in list(logging.root.manager.loggerDict.keys()):
        if name == "legalrag" or name.startswith("legalrag."):
            logging.getLogger(name).setLevel(logging_level)

    if message_only_loggers:
        for name in message_only_loggers:
            lg = logging.getLogger(name)
            lg.setLevel(logging.INFO)
            for h in lg.handlers:
                h.setFormatter(logging.Formatter("%(message)s"))
            lg.propagate = False

    if silence_colbert_stdout:
        try:
            from legalrag.retrieval.colbert_retriever import ColBERTRetriever  
            _orig_init = ColBERTRetriever._init_searcher
            _orig_search = ColBERTRetriever.search

            def _init_quiet(self):
                with suppress_output():
                    return _orig_init(self)

            def _search_quiet(self, *args, **kwargs):
                with suppress_output():
                    return _orig_search(self, *args, **kwargs)

            ColBERTRetriever._init_searcher = _init_quiet  
            ColBERTRetriever.search = _search_quiet   
        except Exception:
            pass

    if silence_tqdm:
        # Many libs respect these toggles.
        os.environ.setdefault("TQDM_DISABLE", "1")

    if silence_hf_progress:
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        os.environ.setdefault("DISABLE_TORCHVISION", "1")
        os.environ.setdefault("TRANSFORMERS_NO_IMAGE_PROCESSING", "1")

    if silence_tensorflow:
        # TensorFlow C++ log level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")

    # If transformers is imported, we can try to disable its verbosity.
    try:
        from transformers.utils import logging as hf_logging   
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


@contextmanager
def suppress_output(*, stdout: bool = True, stderr: bool = True) -> Iterator[None]:
    """Temporarily redirect stdout/stderr to os.devnull.

    Use this sparingly—only for extremely noisy calls (e.g., one-time index builds).
    """
    devnull = open(os.devnull, "w")
    old_stdout: Optional[object] = None
    old_stderr: Optional[object] = None
    try:
        if stdout:
            old_stdout = sys.stdout
            sys.stdout = devnull  
        if stderr:
            old_stderr = sys.stderr
            sys.stderr = devnull  
        yield
    finally:
        try:
            if stdout and old_stdout is not None:
                sys.stdout = old_stdout  
            if stderr and old_stderr is not None:
                sys.stderr = old_stderr  
        finally:
            devnull.close()
