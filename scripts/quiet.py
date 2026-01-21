"""Quiet-mode utilities for notebooks and demos.

Goals:
- Suppress noisy logs (INFO/DEBUG), progress bars, and common framework warnings.
- Keep meaningful outputs (tables/figures/explicit prints) visible.
- Provide an opt-in context manager to silence extremely noisy calls.

Usage (recommended at top of a notebook):
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

    # Per-library loggers (safe even if libs are not imported).
    for name in [
        "jieba",
        "transformers",
        "sentence_transformers",
        "faiss",
        "tensorflow",
        "torch",
        "urllib3",
    ]:
        logging.getLogger(name).setLevel(logging_level)

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
        from transformers.utils import logging as hf_logging  # type: ignore
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
            sys.stdout = devnull  # type: ignore
        if stderr:
            old_stderr = sys.stderr
            sys.stderr = devnull  # type: ignore
        yield
    finally:
        try:
            if stdout and old_stdout is not None:
                sys.stdout = old_stdout  # type: ignore
            if stderr and old_stderr is not None:
                sys.stderr = old_stderr  # type: ignore
        finally:
            devnull.close()
