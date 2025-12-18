import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

# -------- Config -------- 
REPO_ID = "flora-l/lawdata"
ZIP_NAME = "index.zip"
REPO_TYPE = "dataset"

# -------- Paths --------
# This script is executed at repo_root/data 
DATA_DIR = Path.cwd()                     
INDEX_DIR = DATA_DIR / "index"            
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

print(f"[download_index] cwd={DATA_DIR.resolve()}")
print(f"[download_index] index_dir={INDEX_DIR.resolve()}")
print(f"[download_index] repo_id={REPO_ID} repo_type={REPO_TYPE} zip={ZIP_NAME}")

# -------- Download --------
zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type=REPO_TYPE)
zip_path = Path(zip_path).resolve()
print(f"[download_index] downloaded_zip={zip_path}")

# -------- Extract into DATA_DIR --------
with zipfile.ZipFile(str(zip_path), "r") as z:
    z.extractall(str(DATA_DIR))

# Some zips might contain nested "data/index/*" or "index/*"
# Normalize into DATA_DIR/index if needed.
candidates = [
    DATA_DIR / "index",            # index/*
    DATA_DIR / "data" / "index",   # data/index/*
]

# If faiss.index is not in INDEX_DIR yet, try to find it in candidates and copy over
required = ["faiss.index", "faiss_meta.jsonl", "bm25.pkl"]

def has_all(p: Path) -> bool:
    return all((p / name).exists() for name in required)

src = None
if has_all(INDEX_DIR):
    src = INDEX_DIR
else:
    for cand in candidates:
        if cand.exists() and has_all(cand):
            src = cand
            break

if src and src != INDEX_DIR:
    print(f"[download_index] normalize from {src.resolve()} -> {INDEX_DIR.resolve()}")
    for name in required:
        (INDEX_DIR / name).write_bytes((src / name).read_bytes())

# -------- Validate & print absolute paths --------
missing = [name for name in required if not (INDEX_DIR / name).exists()]
print("[download_index] extracted files absolute paths:")
for name in required:
    p = (INDEX_DIR / name).resolve()
    print(f"  - {name}: exists={p.exists()} path={p}")

if missing:
    raise FileNotFoundError(f"Missing index files under {INDEX_DIR.resolve()}: {missing}")

print("[download_index] index ready.")
