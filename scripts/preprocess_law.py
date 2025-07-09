from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from legalrag.config import AppConfig
from legalrag.utils.logger import get_logger
import re

logger = get_logger(__name__)

CN_NUM = r"[一二三四五六七八九十百千万〇零0-9]+"

def normalize_article_no(s: str) -> str:
    if isinstance(s, int):
        return s

    s = (s or "").strip()
    m = re.search(r"(\d+)", s)
    if m:
        return str(int(m.group(1)))

    # 中文数字（覆盖“第五百八十五条”）
    CN_DIGIT = {"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
    CN_UNIT = {"十":10,"百":100,"千":1000}
    CN_BIG  = {"万":10_000,"亿":100_000_000}

    s2 = re.sub(r"[第条\s]", "", s)
    total, section, number = 0, 0, 0
    for ch in s2:
        if ch in CN_DIGIT:
            number = CN_DIGIT[ch]
        elif ch in CN_UNIT:
            unit = CN_UNIT[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        elif ch in CN_BIG:
            big = CN_BIG[ch]
            section += number
            number = 0
            total += section * big
            section = 0
    section += number
    v = total + section
    return str(v) if v > 0 else ""



# ---- Headings (can be "第三编", "第一分编", "第一章", "第一节"; may include wide spaces) ----
PART_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*编(?P<title>.*)$")
SUBPART_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*分编(?P<title>.*)$")
CHAPTER_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*章(?P<title>.*)$")
SECTION_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*节(?P<title>.*)$")

# ---- Article (supports: 第463条 / 463条 / 第四百六十三条 / 第四百六十三条...) ----
ARTICLE_LINE_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*条(?P<rest>.*)$")

# ---- Fallback: scan the whole text, do not require line-start headings ----
# Trigger on newline boundary OR start-of-text; tolerate spaces and optional "第"
ARTICLE_SCAN_RE = re.compile(
    rf"(?m)(?:^|\n)\s*(?:第\s*)?(?P<num>{CN_NUM})\s*条"
)

def _clean_line(s: str) -> str:
    s = s.replace("\u3000", " ")  # full-width space
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def _heading(kind: str, num: str, title: str) -> str:
    title = _clean_line(title)
    return f"{num}{kind} {title}".strip() if title else f"{num}{kind}"

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030", errors="ignore")

def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

@dataclass
class State:
    law_name: str
    source: str
    part: str = ""
    subpart: str = ""
    chapter: str = ""
    section: str = ""
    cur_key: Optional[str] = None
    cur_no: Optional[str] = None
    cur_lines: List[str] = None

    def __post_init__(self) -> None:
        self.cur_lines = self.cur_lines or []

def _finalize(st: State, out: List[Dict]) -> None:
    if not st.cur_no:
        return
    text = "\n".join(st.cur_lines).strip()
    if not text:
        return

    article_key = st.cur_key or st.cur_no
    rec_id = f"{st.source}::{article_key}"

    out.append(
        {
            "id": rec_id,
            "law_name": st.law_name,
            "part": st.part,
            "subpart": st.subpart,
            "chapter": st.chapter,
            "section": st.section,
            "article_no": st.cur_no,     # normalized: 第...条
            "article_key": article_key,  # raw: 四百六十三 
            "article_id":  normalize_article_no(st.cur_no) , # 463
            "text": text,
            "source": st.source,
        }
    )
    st.cur_key = None
    st.cur_no = None
    st.cur_lines = []

def parse_by_lines(text: str, source: str, law_name: str) -> List[Dict]:
    st = State(law_name=law_name, source=source)
    records: List[Dict] = []

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            if st.cur_no:
                st.cur_lines.append("")
            continue

        # headings
        m = PART_RE.match(line)
        if m and not ARTICLE_LINE_RE.match(line):
            st.part = _heading("编", m.group("num"), m.group("title"))
            st.subpart = ""
            st.chapter = ""
            st.section = ""
            continue

        m = SUBPART_RE.match(line)
        if m and not ARTICLE_LINE_RE.match(line):
            st.subpart = _heading("分编", m.group("num"), m.group("title"))
            st.chapter = ""
            st.section = ""
            continue

        m = CHAPTER_RE.match(line)
        if m and not ARTICLE_LINE_RE.match(line):
            st.chapter = _heading("章", m.group("num"), m.group("title"))
            st.section = ""
            continue

        m = SECTION_RE.match(line)
        if m and not ARTICLE_LINE_RE.match(line):
            st.section = _heading("节", m.group("num"), m.group("title"))
            continue

        # article
        m = ARTICLE_LINE_RE.match(line)
        if m:
            _finalize(st, records)
            key = _clean_line(m.group("num"))
            st.cur_key = key
            st.cur_no = f"第{key}条"
            st.cur_lines = [line]
            continue

        # body line
        if st.cur_no:
            st.cur_lines.append(line)

    _finalize(st, records)
    return records

def parse_by_scan_fallback(text: str, source: str, law_name: str) -> List[Dict]:
    """
    Fallback when the input is not cleanly line-broken (common with PDF copy-paste).
    We scan the whole text for article markers and slice segments between them.
    Headings are not reliably recoverable in fallback mode, so we keep them blank.
    """
    matches = list(ARTICLE_SCAN_RE.finditer(text))
    if not matches:
        return []

    records: List[Dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()

        key = _clean_line(m.group("num"))
        article_no = f"第{key}条"
        rec_id = f"{source}::{key}"

        # keep segment compact (normalize spaces) but preserve newlines
        seg_lines = [ _clean_line(x) for x in segment.splitlines() ]
        seg_text = "\n".join([x for x in seg_lines if x != ""]).strip()

        records.append(
            {
                "id": rec_id,
                "law_name": law_name,
                "part": "",
                "subpart": "",
                "chapter": "",
                "section": "",
                "article_no": article_no,
                "article_key": key,
                "article_id":  normalize_article_no(article_no) ,
                "text": seg_text,
                "source": source,
            }
        )
    return records

def debug_preview(text: str) -> None:
    lines = text.splitlines()
    logger.info("---- raw preview (first 20 lines) ----")
    for i, ln in enumerate(lines[:20], start=1):
        logger.info(f"{i:02d}: {ln[:200]}")
    logger.info("-------------------------------------")

    # quick counts
    line_hits = sum(1 for ln in lines if ARTICLE_LINE_RE.match(_clean_line(ln)))
    scan_hits = len(list(ARTICLE_SCAN_RE.finditer(text)))
    logger.info(f"[debug] ARTICLE_LINE_RE line hits = {line_hits}")
    logger.info(f"[debug] ARTICLE_SCAN_RE scan hits = {scan_hits}")

def main() -> int:
    cfg = AppConfig.load(None)

    raw_dir = Path(cfg.paths.raw_dir)
    out_path = Path(cfg.paths.contract_law_jsonl)

    txt_files = sorted(raw_dir.glob("*.txt"))
    logger.info(f"Raw dir: {raw_dir.resolve()}")
    logger.info(f"Found txt files: {[p.name for p in txt_files]}")

    if not txt_files:
        logger.error(f"No .txt found under {raw_dir}")
        return 2

    all_records: List[Dict] = []
    for p in txt_files:
        logger.info(f"Parsing: {p}")
        text = _read_text(p)
        logger.info(f"File size: {len(text)} chars")

        recs = parse_by_lines(text, source=p.name, law_name="中华人民共和国民法典")
        if len(recs) == 0:
            logger.warning("Line-based parse got 0 records; switching to scan fallback.")
            recs = parse_by_scan_fallback(text, source=p.name, law_name="中华人民共和国民法典")

        logger.info(f"Parsed records from {p.name}: {len(recs)}")
        all_records.extend(recs)

    logger.info(f"Total records: {len(all_records)}")
    _write_jsonl(out_path, all_records)
    logger.info(f"Saved JSONL to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

