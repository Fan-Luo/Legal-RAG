from __future__ import annotations
import uuid
from dataclasses import dataclass
from pathlib import Path
from fastapi import UploadFile
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class IngestResult:
    doc_id: str
    jsonl_path: str
    num_chunks: int

class PDFIngestor:
    def __init__(self, cfg):
        self.cfg = cfg

    async def ingest(self, file: UploadFile) -> IngestResult:
        doc_id = uuid.uuid4().hex
        out = Path(self.cfg.paths.processed_dir) / f"{doc_id}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        logger.info("PDF ingested doc_id=%s -> %s", doc_id, out)
        return IngestResult(doc_id=doc_id, jsonl_path=str(out), num_chunks=0)
