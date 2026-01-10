import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# -------- Config --------
REPO_ID = "flora-l/lawdata"
ZIP_NAME = "data.zip"          
REPO_TYPE = "dataset"

# -------- Paths --------
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"[download_data] data dir={DATA_DIR.resolve()}")
print(f"[download_data] repo_id={REPO_ID} repo_type={REPO_TYPE} zip={ZIP_NAME}")

# -------- Download --------
zip_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=ZIP_NAME,
    repo_type=REPO_TYPE
)
zip_path = Path(zip_path).resolve()
print(f"[download_data] downloaded_zip={zip_path}")

# -------- Extract --------
with zipfile.ZipFile(str(zip_path), "r") as z:
    for member in z.infolist():
        target_path = DATA_DIR / member.filename

        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member, "r") as src, open(target_path, "wb") as dst:
            dst.write(src.read())
        print(f"[download_data] wrote: {target_path}")

print("[download_data] done. All files merged into data/.")