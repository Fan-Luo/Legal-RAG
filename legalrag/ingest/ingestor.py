from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.pdf.parser import extract_text_from_pdf
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------
# Helpers
# ---------------------------
_WHITESPACE_RE = re.compile(r"[ \t\u00A0]+")


def _normalize_text(text: str) -> str:
    """
    Normalize raw extracted text from PDF/OCR:
    - normalize whitespace
    - normalize newlines
    - remove superfluous blank lines
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    # remove trailing spaces on lines
    text = "\n".join(line.strip() for line in text.split("\n"))
    # collapse multiple blank lines
    lines: List[str] = []
    blank = 0
    for line in text.split("\n"):
        if not line:
            blank += 1
            if blank <= 1:
                lines.append("")
        else:
            blank = 0
            lines.append(line)
    return "\n".join(lines).strip()


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _make_doc_id(source_name: str, raw_text: str) -> str:
    """
    Stable doc_id for reproducibility.
    """
    base = f"{source_name}|{_sha1(raw_text)[:12]}"
    return _sha1(base)[:16]


def _iter_paragraphs(text: str) -> Iterable[Tuple[str, int, int]]:
    """
    Yield (para_text, start_char, end_char) from normalized text.
    """
    if not text:
        return
    start = 0
    for m in re.finditer(r"\n{2,}", text):
        end = m.start()
        para = text[start:end].strip()
        if para:
            yield para, start, end
        start = m.end()
    tail = text[start:].strip()
    if tail:
        yield tail, start, len(text)


def _chunk_by_tokens_like(text: str, target_chars: int = 600, overlap_chars: int = 80) -> List[Tuple[str, int, int]]:
    """
    Light-weight chunker without tokenizer dependency.
    - chunk roughly by characters, prefer sentence boundaries.
    - returns list of (chunk_text, start_char, end_char)
    """
    if not text:
        return []

    # split by punctuation for Chinese/English
    sents = re.split(r"(?<=[。！？!?；;])", text)
    sents = [s.strip() for s in sents if s.strip()]
    chunks: List[Tuple[str, int, int]] = []

    cursor = 0
    buf: List[str] = []
    buf_len = 0
    chunk_start = 0

    for sent in sents:
        if not buf:
            chunk_start = cursor

        buf.append(sent)
        buf_len += len(sent)

        cursor += len(sent)

        if buf_len >= target_chars:
            chunk_text = "".join(buf).strip()
            chunk_end = chunk_start + len(chunk_text)
            chunks.append((chunk_text, chunk_start, chunk_end))

            # overlap: keep tail
            if overlap_chars > 0 and len(chunk_text) > overlap_chars:
                tail = chunk_text[-overlap_chars:]
                buf = [tail]
                buf_len = len(tail)
                chunk_start = chunk_end - overlap_chars
            else:
                buf = []
                buf_len = 0

    if buf:
        chunk_text = "".join(buf).strip()
        chunk_end = chunk_start + len(chunk_text)
        chunks.append((chunk_text, chunk_start, chunk_end))

    return chunks


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------
# Ingestor
# ---------------------------
@dataclass
class IngestResult:
    doc_id: str
    source: str
    text_len: int
    num_chunks: int
    jsonl_path: Optional[str] = None
    preview: str = ""


class PDFIngestor:
    """
    PDF -> extract text (OCR fallback) -> normalize -> chunk -> LawChunk -> persist JSONL
    Output chunks are stored as LawChunk with:
        - article_id: doc-local chunk id (stable)
        - article_no: display label, e.g. "DocChunk-0003"
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    async def ingest(self, file):
        """Adapter: FastAPI UploadFile -> temp PDF -> ingest_pdf_to_jsonl.
        This method only adapts I/O; it preserves IngestResult semantics.
        """
        from pathlib import Path
        import tempfile
        from fastapi import UploadFile

        if not isinstance(file, UploadFile):
            raise TypeError("file must be fastapi.UploadFile")

        suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
        tmp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(await file.read())

            # reset stream for any downstream use
            try:
                await file.seek(0)
            except Exception:
                pass

            return self.ingest_pdf_to_jsonl(tmp_path)

        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def ingest_pdf_to_jsonl(
        self,
        pdf_path: str | Path,
        *,
        law_name: str = "PDF_CASE_DOC",
        out_jsonl: Optional[str | Path] = None,
        chunk_chars: int = 650,
        overlap_chars: int = 90,
    ) -> IngestResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        t0 = time.time()
        raw_text = extract_text_from_pdf(pdf_path, self.cfg)
        norm_text = _normalize_text(raw_text)

        source_name = pdf_path.name
        doc_id = _make_doc_id(source_name, raw_text)

        logger.info(f"[Ingest] doc_id={doc_id} source={source_name} text_len={len(norm_text)}")

        chunks: List[LawChunk] = []
        idx = 0

        # Prefer paragraph boundary first; then chunk inside each paragraph
        for para, para_s, _para_e in _iter_paragraphs(norm_text):
            sub_chunks = _chunk_by_tokens_like(para, target_chars=chunk_chars, overlap_chars=overlap_chars)
            for c_text, c_s, c_e in sub_chunks:
                idx += 1
                chunk_id = f"{doc_id}:{idx:04d}"  # stable doc-local id

                chunks.append(
                    LawChunk(
                        id=chunk_id,
                        law_name=law_name,
                        chapter=None,
                        section=None,
                        article_no=f"DocChunk-{idx:04d}",
                        article_id=chunk_id,
                        text=c_text,
                        source=str(pdf_path),
                        start_char=int(para_s + c_s),
                        end_char=int(para_s + c_e),
                    )
                )

        # output path
        if out_jsonl is None:
            out_jsonl = Path(self.cfg.paths.processed_dir) / f"ingested_{doc_id}.jsonl"
        else:
            out_jsonl = Path(out_jsonl)

        n = _write_jsonl(out_jsonl, (c.model_dump() for c in chunks))

        logger.info(f"[Ingest] saved {n} chunks -> {out_jsonl} (t={time.time()-t0:.2f}s)")

        preview = norm_text[:800].replace("\n", " ")
        return IngestResult(
            doc_id=doc_id,
            source=source_name,
            text_len=len(norm_text),
            num_chunks=n,
            jsonl_path=str(out_jsonl),
            preview=preview,
        )
