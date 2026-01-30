from __future__ import annotations

import re

_RE_ZH = re.compile(r"[\u4e00-\u9fff]")
_RE_EN = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    """Return 'zh' or 'en' (best-effort)."""
    if not text:
        return "zh"
    zh = len(_RE_ZH.findall(text))
    en = len(_RE_EN.findall(text))
    return "en" if en > zh else "zh"
