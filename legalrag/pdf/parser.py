from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import List, Tuple

import pdfplumber

from legalrag.config import AppConfig
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from pdf2image import convert_from_path
    import pytesseract

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
    logger.warning("pdf2image/pytesseract 不可用，扫描 PDF OCR 将被禁用。")


def _clean_line_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_header_footer_key(text: str) -> str:
    text = _clean_line_text(text)
    text = re.sub(r"\d+", "", text)
    text = text.replace("第", "").replace("页", "")
    return text.strip()

_PART_RE = re.compile(r"第[一二三四五六七八九十百千万〇零0-9]+编")
_SUBPART_RE = re.compile(r"第[一二三四五六七八九十百千万〇零0-9]+分编")
_CHAPTER_RE = re.compile(r"第[一二三四五六七八九十百千万〇零0-9]+章")
_SECTION_RE = re.compile(r"第[一二三四五六七八九十百千万〇零0-9]+节")
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千万〇零0-9]+条")
_EN_SECTION_RE = re.compile(r"^§\s*[0-9A-Za-z-]+\.?")
_EN_ARTICLE_RE = re.compile(r"^ARTICLE\s+[0-9A-Za-z-]+\s*[-–—]\s*.+$", re.IGNORECASE)
_EN_PART_RE = re.compile(r"^PART\s+[0-9A-Za-z-]+\.?\s*.+$", re.IGNORECASE)
_FOOTER_RE = re.compile(
    r"(ICP备|打印本稿|来源：|版权所有：)"
)
_TOC_RE = re.compile(r"^(目录|目\s*录|contents|table\s+of\s+contents)$", re.IGNORECASE)
_TOC_DOTS_RE = re.compile(r"\.{2,}\s*\d+\s*$")
_TOC_TRAIL_PAGE_RE = re.compile(r"\s+\d+\s*$")


def _looks_like_toc_line(line: str) -> bool:
    if not line:
        return False
    # If an article line already includes body text, treat as正文.
    if _ARTICLE_RE.search(line) and re.search(r"[。！？；，,]", line):
        return False
    if (
        _PART_RE.search(line)
        or _SUBPART_RE.search(line)
        or _CHAPTER_RE.search(line)
        or _SECTION_RE.search(line)
        or _ARTICLE_RE.search(line)
    ):
        return True
    if _EN_SECTION_RE.match(line) or _EN_ARTICLE_RE.match(line) or _EN_PART_RE.match(line):
        return True
    if _TOC_DOTS_RE.search(line):
        return True
    if _TOC_TRAIL_PAGE_RE.search(line) and (_CHAPTER_RE.search(line) or _ARTICLE_RE.search(line)):
        return True
    return False


def _find_toc_cut_pos(text: str) -> int | None:
    lines = text.splitlines()
    if not lines:
        return None
    starts: List[int] = []
    pos = 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1

    for i, raw in enumerate(lines):
        if _TOC_RE.match(_clean_line_text(raw)):
            j = i + 1
            while j < len(lines):
                line = _clean_line_text(lines[j])
                if not line:
                    j += 1
                    continue
                if _looks_like_toc_line(line):
                    # If this is a heading and the next non-empty line is正文条文，return heading.
                    if _PART_RE.search(line) or _SUBPART_RE.search(line) or _CHAPTER_RE.search(line) or _SECTION_RE.search(line):
                        k = j + 1
                        while k < len(lines):
                            nxt = _clean_line_text(lines[k])
                            if not nxt:
                                k += 1
                                continue
                            if _ARTICLE_RE.search(nxt) and re.search(r"[。！？；，,]", nxt):
                                return starts[j]
                            break
                    j += 1
                    continue
                return starts[j]
            return None
    return None


def _find_prev_heading_start(text: str, start_idx: int) -> int | None:
    lines = text.splitlines()
    if not lines:
        return None
    pos = 0
    last_heading_pos = None
    for ln in lines:
        line_start = pos
        line_end = pos + len(ln)
        if line_end >= start_idx:
            break
        clean = _clean_line_text(ln)
        if clean and (
            _PART_RE.match(clean)
            or _SUBPART_RE.match(clean)
            or _CHAPTER_RE.match(clean)
            or _SECTION_RE.match(clean)
        ):
            last_heading_pos = line_start
        pos = line_end + 1
    return last_heading_pos


def _normalize_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00A0]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
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


def _trim_law_body(text: str) -> str:
    if not text:
        return ""
    text = _normalize_pdf_text(text)
    start_idx = None
    toc_cut = _find_toc_cut_pos(text)
    if toc_cut is not None:
        text_after = text[toc_cut:]
        m_ch = _CHAPTER_RE.search(text_after)
        m_ar = _ARTICLE_RE.search(text_after)
        m_en = _EN_SECTION_RE.search(text_after) or _EN_ARTICLE_RE.search(text_after) or _EN_PART_RE.search(text_after)
        starts = [m.start() for m in (m_ch, m_ar, m_en) if m]
        if starts:
            start_idx = toc_cut + min(starts)
            heading_start = _find_prev_heading_start(text_after, start_idx - toc_cut)
            if heading_start is not None:
                start_idx = toc_cut + heading_start
    else:
        m_ch = _CHAPTER_RE.search(text)
        m_ar = _ARTICLE_RE.search(text)
        m_en = _EN_SECTION_RE.search(text) or _EN_ARTICLE_RE.search(text) or _EN_PART_RE.search(text)
        starts = [m.start() for m in (m_ch, m_ar, m_en) if m]
        if starts:
            start_idx = min(starts)
            heading_start = _find_prev_heading_start(text, start_idx)
            if heading_start is not None:
                start_idx = heading_start
    if start_idx is not None and start_idx > 0:
        text = text[start_idx:]
    last_article = None
    for m in _ARTICLE_RE.finditer(text):
        last_article = m
    if last_article:
        m_footer = _FOOTER_RE.search(text, pos=last_article.start())
        if m_footer:
            text = text[: m_footer.start()]
    return text.strip()

def _page_lines_with_pos(page) -> List[Tuple[str, float, float]]:
    words = page.extract_words(
        x_tolerance=2,
        y_tolerance=2,
        keep_blank_chars=False,
        use_text_flow=True,
    )
    if not words:
        return []

    lines: List[Tuple[str, float, float]] = []
    words = sorted(words, key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))
    cur_top = None
    cur_bottom = None
    cur_parts: List[Tuple[str, float, float]] = []

    def flush_line():
        if not cur_parts:
            return
        parts = sorted(cur_parts, key=lambda x: x[1])
        line = ""
        prev_x1 = None
        for t, x0, x1 in parts:
            if prev_x1 is not None and x0 - prev_x1 > 1:
                line += " "
            line += t
            prev_x1 = x1
        lines.append((_clean_line_text(line), float(cur_top or 0.0), float(cur_bottom or 0.0)))

    for w in words:
        top = float(w.get("top", 0.0))
        bottom = float(w.get("bottom", 0.0))
        text = str(w.get("text") or "").strip()
        if not text:
            continue
        if cur_top is None:
            cur_top = top
            cur_bottom = bottom
        if abs(top - cur_top) <= 2:
            cur_bottom = max(cur_bottom or bottom, bottom)
            cur_parts.append((text, float(w.get("x0", 0.0)), float(w.get("x1", 0.0))))
        else:
            flush_line()
            cur_top = top
            cur_bottom = bottom
            cur_parts = [(text, float(w.get("x0", 0.0)), float(w.get("x1", 0.0)))]
    flush_line()
    return [ln for ln in lines if ln[0]]


def _extract_text_docling(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        return ""

    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = getattr(result, "document", None) or result
        for attr in ("export_to_text", "export_to_markdown"):
            fn = getattr(doc, attr, None)
            if callable(fn):
                return fn()
    except Exception as exc:
        logger.info("[PDF] docling extract failed: %s", exc)
    return ""


def extract_docling_blocks(path: str | Path) -> List[str]:
    path = Path(path)
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        return []

    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = getattr(result, "document", None) or result
    except Exception as exc:
        logger.info("[PDF] docling block extract failed: %s", exc)
        return []

    blocks: List[str] = []
    elements = getattr(doc, "elements", None)
    if elements:
        for el in elements:
            txt = getattr(el, "text", None) or getattr(el, "content", None)
            if txt and str(txt).strip():
                blocks.append(str(txt).strip())

    if not blocks:
        for attr in ("export_to_text", "export_to_markdown"):
            fn = getattr(doc, attr, None)
            if callable(fn):
                text = fn()
                if text and str(text).strip():
                    blocks = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
                break

    return blocks


def _extract_text_layout(path: Path) -> str:
    pages_lines: List[List[Tuple[str, float, float]]] = []
    header_counts = {}
    footer_counts = {}

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            lines = _page_lines_with_pos(page)
            pages_lines.append(lines)
            if not lines:
                continue
            h = float(page.height or 0.0)
            header_cutoff = h * 0.12
            footer_cutoff = h * 0.88
            for text, top, bottom in lines:
                key = _normalize_header_footer_key(text)
                if not key:
                    continue
                if top <= header_cutoff:
                    header_counts[key] = header_counts.get(key, 0) + 1
                if bottom >= footer_cutoff:
                    footer_counts[key] = footer_counts.get(key, 0) + 1

    if not pages_lines:
        return ""

    min_repeat = max(2, int(len(pages_lines) * 0.3))
    header_keys = {k for k, v in header_counts.items() if v >= min_repeat}
    footer_keys = {k for k, v in footer_counts.items() if v >= min_repeat}

    cleaned_pages = []
    for lines in pages_lines:
        out_lines = []
        for text, top, bottom in lines:
            key = _normalize_header_footer_key(text)
            if key and (key in header_keys or key in footer_keys):
                continue
            if re.fullmatch(r"\d+", text):
                continue
            out_lines.append(text)
        cleaned_pages.append("\n".join(out_lines).strip())

    return "\n\n".join([p for p in cleaned_pages if p])


def extract_text_from_pdf(path: str | Path, cfg: AppConfig) -> str:
    path = Path(path)
    if getattr(cfg.pdf, "use_docling", False):
        dl_text = _extract_text_docling(path)
        if dl_text and dl_text.strip():
            logger.info("[PDF] docling extraction used (len=%d)", len(dl_text))
            return _trim_law_body(dl_text)

    texts = []
    ocr_pages = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
            else:
                ocr_pages.append(i)

    if ocr_pages and cfg.pdf.enable_ocr and OCR_AVAILABLE:
        logger.info(f"[PDF] {len(ocr_pages)} 页可能为扫描件，尝试 OCR")
        images = convert_from_path(str(path))
        for i in ocr_pages:
            img = images[i]
            txt = pytesseract.image_to_string(img, lang=cfg.pdf.ocr_lang)
            texts.append(txt)

    raw_text = "\n\n".join(texts)
    layout_text = _extract_text_layout(path) if getattr(cfg.pdf, "use_layout_extraction", True) else ""
    if layout_text and len(layout_text) >= len(raw_text) * 0.6:
        logger.info(
            "[PDF] layout-aware extraction used (layout_len=%d raw_len=%d)",
            len(layout_text),
            len(raw_text),
        )
        return _trim_law_body(layout_text)
    if layout_text:
        logger.info(
            "[PDF] layout-aware extraction fallback to raw (layout_len=%d raw_len=%d)",
            len(layout_text),
            len(raw_text),
        )
    return _trim_law_body(raw_text)
