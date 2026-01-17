from __future__ import annotations

from pathlib import Path
import re
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
            return dl_text

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
        return layout_text
    if layout_text:
        logger.info(
            "[PDF] layout-aware extraction fallback to raw (layout_len=%d raw_len=%d)",
            len(layout_text),
            len(raw_text),
        )
    return raw_text
