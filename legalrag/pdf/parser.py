from __future__ import annotations

from pathlib import Path

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


def extract_text_from_pdf(path: str | Path, cfg: AppConfig) -> str:
    path = Path(path)
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

    full_text = "\n\n".join(texts)
    return full_text
