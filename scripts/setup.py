import logging
from pathlib import Path
import os, subprocess
import sys
import torch
from scripts.quiet import install_quiet

def setup_logging(
    level=logging.INFO,
    fmt="%(message)s",
    scope="legalrag",
):
    """
    Configure logging for the LegalRAG repository.

    This function removes existing handlers attached to loggers
    under the given scope and installs a clean StreamHandler.
    """
    base = logging.getLogger(scope)
    base.setLevel(level)

    # Remove all handlers under this scope
    for name, logger in logging.root.manager.loggerDict.items():
        if name == scope or name.startswith(scope + "."):
            lg = logging.getLogger(name)
            for h in lg.handlers[:]:
                lg.removeHandler(h)
            lg.propagate = False

    # Attach a single handler to base logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    base.addHandler(handler)

def run(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        tail="\n".join(res.stdout.splitlines()[-40:])
        raise RuntimeError(f"Command failed: {cmd}\n--- output tail ---\n{tail}")
    return res.stdout

install_quiet()
setup_logging()
run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])

# Option 1: load preprocessed law data

run([sys.executable, "-m", "data.download_data"])

torch.cuda.empty_cache()

# Option 2:  run the offline preprocessing to convert raw legal texts in data/raw/ into normalized data artifacts, retrieval indices, and a legal knowledge graph through the following steps:
# from pathlib import Path
# import os

# try:
#     !python -m scripts.preprocess_law
# except SystemExit:
#     print("Preprocessing completed.")

# try:
#     !python -m scripts.build_index
# except SystemExit:
#     print("Index building completed.")

# try:
#     !python -m scripts.build_graph
# except SystemExit:
#     print("Law graph building completed.") 



