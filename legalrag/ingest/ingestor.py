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
from legalrag.pdf.parser import extract_text_from_pdf, extract_docling_blocks
from legalrag.utils.lang import detect_lang
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


def _short_name(name: str, max_len: int = 12) -> str:
    base = (name or "").strip()
    if not base:
        return "upload"
    base = re.sub(r"\s+", "", base)
    return base[:max_len]


def _has_zh_or_en(text: str) -> bool:
    if not text:
        return False
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return True
    return any("a" <= ch.lower() <= "z" for ch in text)


def _extract_title_from_text(text: str) -> str:
    for line in (text or "").splitlines():
        t = line.strip()
        if t:
            return t
    return ""


def _shorten_title(title: str, max_len: int = 20) -> str:
    if not title:
        return "upload"
    if len(title) <= max_len:
        return title
    return f"{title[:10]}***{title[-10:]}"


_TITLE_HEADING_RE = re.compile(
    r"^\s*("
    r"(第[一二三四五六七八九十百千万〇零0-9]+(编|分编|章|节|条))"
    r"|((ARTICLE|PART)\s+[0-9A-Za-z-]+)"
    r")"
)


def _is_heading_like_title(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    return bool(_TITLE_HEADING_RE.match(t))


CN_NUM = r"[一二三四五六七八九十百千万〇零0-9]+"
ARTICLE_LINE_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*条(?P<rest>.*)$")

def _extract_label_from_chunk(text: str) -> Optional[str]:
    head = (text or "").strip()
    if not head:
        return None
    head = head.splitlines()[0].strip()
    # Prefer explicit article markers.
    m = ARTICLE_LINE_RE.match(head)
    if m:
        num = re.sub(r"[ \t]", "", m.group("num"))
        return f"第{num}条"
    # Fallback: take prefix before first punctuation as a label.
    cut = re.split(r"[，,。；;:：\\s]", head, maxsplit=1)[0].strip()
    return cut or None

def _make_unique_label(label: str, seen: Dict[str, int]) -> str:
    if not label:
        return label
    count = seen.get(label, 0) + 1
    seen[label] = count
    return label if count == 1 else f"{label}-{count}"


def _parse_law_records_from_blocks(blocks: List[str], source: str, law_name: str) -> List[Dict[str, Any]]:
    try:
        from scripts.preprocess_law import normalize_article_no
    except Exception:
        normalize_article_no = lambda s: ""

    records: List[Dict[str, Any]] = []
    cur_no = None
    cur_key = None
    cur_lines: List[str] = []

    def finalize():
        nonlocal cur_no, cur_key, cur_lines
        if not cur_no:
            return
        text = "\n".join([ln for ln in cur_lines if ln.strip()]).strip()
        if not text:
            cur_no = None
            cur_key = None
            cur_lines = []
            return
        rec_id = f"{source}::{cur_key or cur_no}"
        records.append(
            {
                "id": rec_id,
                "law_name": law_name,
                "part": "",
                "subpart": "",
                "chapter": "",
                "section": "",
                "article_no": cur_no,
                "article_key": cur_key,
                "article_id": normalize_article_no(cur_no),
                "text": text,
                "source": source,
            }
        )
        cur_no = None
        cur_key = None
        cur_lines = []

    for raw in blocks:
        line = _normalize_text(str(raw)).strip()
        if not line:
            continue
        m = ARTICLE_LINE_RE.match(line)
        if m:
            finalize()
            key = re.sub(r"[ \t]", "", m.group("num"))
            cur_key = key
            cur_no = f"第{key}条"
            cur_lines = [line]
            continue
        if cur_no:
            cur_lines.append(line)

    finalize()
    return records


def _parse_law_records(text: str, source: str, law_name: str) -> List[Dict[str, Any]]:
    try:
        from scripts.preprocess_law import parse_by_lines, parse_by_scan_fallback
    except Exception:
        return []
    recs_line = parse_by_lines(text, source=source, law_name=law_name)
    recs_scan = parse_by_scan_fallback(text, source=source, law_name=law_name)
    if recs_scan and (len(recs_line) < 10 or len(recs_scan) > len(recs_line)):
        logger.info(
            "[Ingest] parse_by_lines=%d parse_by_scan=%d -> using scan",
            len(recs_line),
            len(recs_scan),
        )
        return recs_scan
    if recs_line:
        logger.info(
            "[Ingest] parse_by_lines=%d parse_by_scan=%d -> using line",
            len(recs_line),
            len(recs_scan),
        )
    return recs_line


def _parse_quality(records: List[Dict[str, Any]], text_len: int) -> Dict[str, Any]:
    total_text = sum(len(str(rec.get("text") or "")) for rec in records)
    coverage = total_text / max(1, text_len)
    avg_len_ratio = (total_text / max(1, len(records))) / max(1, text_len)
    nums = []
    for rec in records:
        val = str(rec.get("article_id") or "")
        if val.isdigit():
            nums.append(int(val))
    nums_sorted = sorted(set(nums))
    gap_ratio = None
    if len(nums_sorted) >= 5:
        span = nums_sorted[-1] - nums_sorted[0] + 1
        if span > 0:
            gap_ratio = 1 - (len(nums_sorted) / span)
    return {
        "coverage": coverage,
        "count": len(records),
        "gap_ratio": gap_ratio,
        "avg_len_ratio": avg_len_ratio,
    }


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
        from fastapi import UploadFile as FastAPIUploadFile
        from starlette.datastructures import UploadFile as StarletteUploadFile

        if file is None:
            raise TypeError("file is required")
        if not isinstance(file, (FastAPIUploadFile, StarletteUploadFile)):
            if not callable(getattr(file, "read", None)):
                raise TypeError("file must be UploadFile-like with read()")
        logger.info(
            "[ingest] upload type=%s filename=%s",
            type(file).__name__,
            getattr(file, "filename", None),
        )

        suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
        orig_name = (file.filename or "upload.pdf")
        short_name = _short_name(Path(orig_name).stem)
        display_name = f"{short_name}"
        tmp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = Path(tmp.name)
                try:
                    data = await file.read()
                except TypeError:
                    data = file.read()
                logger.info("[ingest] read_bytes=%d tmp_path=%s", len(data or b""), tmp_path)
                tmp.write(data)

            # reset stream for any downstream use
            try:
                await file.seek(0)
            except Exception:
                try:
                    file.seek(0)
                except Exception:
                    pass

            logger.info("[ingest] saved upload to %s", tmp_path)
            return self.ingest_pdf_to_jsonl(tmp_path, source_name=orig_name, law_name=display_name)

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
        source_name: Optional[str] = None,
        law_name: str = "PDF_CASE_DOC",
        out_jsonl: Optional[str | Path] = None,
        chunk_chars: int = 650,
        overlap_chars: int = 90,
    ) -> IngestResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        t0 = time.time()
        docling_blocks = []
        if getattr(self.cfg.pdf, "use_docling", False):
            docling_blocks = extract_docling_blocks(pdf_path)
            if docling_blocks:
                logger.info("[Ingest] docling blocks=%d from %s", len(docling_blocks), pdf_path.name)

        raw_text = extract_text_from_pdf(pdf_path, self.cfg)
        norm_text = _normalize_text(raw_text)

        source_name = source_name or pdf_path.name
        doc_id = _make_doc_id(source_name, raw_text)
        display_name = law_name
        stem = Path(source_name).stem
        if not _has_zh_or_en(stem):
            title = _extract_title_from_text(raw_text)
            if not _is_heading_like_title(title):
                display_name = _shorten_title(title)

        doc_lang = detect_lang(norm_text)
        logger.info(f"[Ingest] doc_id={doc_id} source={source_name} text_len={len(norm_text)}")

        chunks: List[LawChunk] = []
        law_records = []
        if docling_blocks:
            law_records = _parse_law_records_from_blocks(docling_blocks, source=source_name, law_name=display_name)
            if not law_records:
                law_records = _parse_law_records("\n".join(docling_blocks), source=source_name, law_name=display_name)
        if not law_records:
            law_records = _parse_law_records(norm_text, source=source_name, law_name=display_name)
        if law_records:
            qc = _parse_quality(law_records, len(norm_text))
            min_records = 20
            min_coverage = 0.3
            max_gap_ratio = 0.5
            max_avg_len_ratio = 0.12
            gap_ratio = qc.get("gap_ratio")
            if (
                qc["count"] < min_records
                or qc["coverage"] < min_coverage
                or (gap_ratio is not None and gap_ratio > max_gap_ratio)
                or qc["avg_len_ratio"] > max_avg_len_ratio
            ):
                logger.info(
                    "[Ingest] law parse low quality (records=%d coverage=%.2f gap_ratio=%s avg_len_ratio=%.3f), fallback to chunking: %s",
                    qc["count"],
                    qc["coverage"],
                    f"{gap_ratio:.2f}" if gap_ratio is not None else "n/a",
                    qc["avg_len_ratio"],
                    source_name,
                )
                law_records = []
            else:
                logger.info("[Ingest] parsed law records=%d from %s", len(law_records), source_name)
                for rec in law_records:
                    chunks.append(
                        LawChunk(
                            id=str(rec.get("id") or ""),
                            law_name=str(rec.get("law_name") or display_name),
                            chapter=rec.get("chapter") or None,
                            section=rec.get("section") or None,
                            article_no=str(rec.get("article_no") or ""),
                            article_id=str(rec.get("article_id") or ""),
                            text=str(rec.get("text") or ""),
                            lang=doc_lang,
                            source=str(rec.get("source") or source_name),
                            start_char=None,
                            end_char=None,
                        )
                    )

        if not law_records:
            logger.info("[Ingest] no usable article markers, fallback to chunking: %s", source_name)
            idx = 0
            label_counts: Dict[str, int] = {}
            # Prefer paragraph boundary first; then chunk inside each paragraph
            for para, para_s, _para_e in _iter_paragraphs(norm_text):
                sub_chunks = _chunk_by_tokens_like(para, target_chars=chunk_chars, overlap_chars=overlap_chars)
                for c_text, c_s, c_e in sub_chunks:
                    idx += 1
                    chunk_id = f"{doc_id}:{idx:04d}"  # stable doc-local id
                    base_label = _extract_label_from_chunk(c_text) or f"{display_name}-{idx:04d}"
                    article_label = _make_unique_label(base_label, label_counts)

                    chunks.append(
                        LawChunk(
                            id=chunk_id,
                            law_name=display_name,
                            chapter=None,
                            section=None,
                            article_no=article_label,
                            article_id=article_label,
                            text=c_text,
                            lang=doc_lang,
                            source=str(source_name),
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
