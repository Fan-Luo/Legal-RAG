from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from legalrag.config import AppConfig
from legalrag.utils.lang import detect_lang
from legalrag.utils.logger import get_logger

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
PART_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*编(?P<title>.*)$", re.M)
SUBPART_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*分编(?P<title>.*)$", re.M)
CHAPTER_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*章(?P<title>.*)$", re.M)
SECTION_RE = re.compile(rf"^\s*(?:第\s*)?(?P<num>{CN_NUM})\s*节(?P<title>.*)$", re.M)
INLINE_HEADING_RE = re.compile(rf"(?:第\s*)?(?P<num>{CN_NUM})\s*(?:分编|编|章|节)\s*.+$")

# ---- Article (supports: 第463条 / 第四百六十三条 / 第四百六十三条...) ----
ARTICLE_LINE_RE = re.compile(rf"^\s*第\s*(?P<num>{CN_NUM})\s*条(?P<rest>.*)$")
ARTICLE_LINE_NO_DAI_RE = re.compile(rf"^\s*(?P<num>{CN_NUM})\s*条(?P<rest>.*)$")

# ---- UCC Section (English) ----
EN_SECTION_LINE_RE = re.compile(r"^\s*§\s*(?P<id>[0-9A-Za-z-]+)\.?\s*(?P<rest>.*)$")
EN_SECTION_SCAN_RE = re.compile(r"(?m)^\s*§\s*(?P<id>[0-9A-Za-z-]+)\.")
EN_ARTICLE_RE = re.compile(r"^\s*ARTICLE\s+(?P<num>[0-9A-Za-z-]+)\s*[-–—]\s*(?P<title>.*)$", re.IGNORECASE)
EN_PART_RE = re.compile(r"^\s*PART\s+(?P<num>[0-9A-Za-z-]+)\.?\s*(?P<title>.*)$", re.IGNORECASE)

# ---- Fallback: scan the whole text, do not require line-start headings ----
# Trigger on newline boundary OR start-of-text; tolerate spaces and optional "第"
ARTICLE_SCAN_RE = re.compile(
    r"(?m)(?<![一二三四五六七八九十百千万〇零0-9])第\s*(?P<num>[一二三四五六七八九十百千万〇零0-9]+)\s*条"
)
ARTICLE_SCAN_NO_DAI_RE = re.compile(
    rf"(?m)(^|\n)\s*(?P<num>{CN_NUM})\s*条"
)

_CITATION_PREFIXES = ("本法", "本章", "本节", "本条例", "本编", "本分编", "依照", "根据")

def _is_citation_start(text: str, start: int) -> bool:
    prefix = text[max(0, start - 6):start]
    return any(prefix.endswith(p) for p in _CITATION_PREFIXES)

def _normalize_article_markers(text: str) -> str:
    if not text:
        return text
    # Join broken article markers across line breaks, e.g., "第十\n三条" -> "第十三条"
    text = re.sub(
        rf"(第\s*{CN_NUM})\s*\n\s*({CN_NUM})\s*条",
        r"\1\2条",
        text,
    )
    text = re.sub(
        rf"(第\s*{CN_NUM})\s*\n\s*条",
        r"\1条",
        text,
    )
    return text

def _clean_line(s: str) -> str:
    s = s.replace("\u3000", " ")  # full-width space
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def _should_break(prev: str, nxt: str) -> bool:
    if not prev:
        return True
    if not nxt:
        return True
    if re.search(r"[。！？；：:]$", prev):
        return True
    if re.match(r"^(第\s*[一二三四五六七八九十百千万〇零0-9]+\s*条)", nxt):
        return True
    if PART_RE.match(nxt) or SUBPART_RE.match(nxt) or CHAPTER_RE.match(nxt) or SECTION_RE.match(nxt):
        return True
    if re.match(r"^[（(]?[一二三四五六七八九十0-9]+[)）\.、]", nxt):
        return True
    return False

def _merge_lines(lines: List[str]) -> str:
    out: List[str] = []
    cur = ""
    for raw in lines:
        line = _clean_line(raw)
        if not line:
            if cur:
                out.append(cur)
                cur = ""
            continue
        if not cur:
            cur = line
            continue
        if _should_break(cur, line):
            out.append(cur)
            cur = line
        else:
            cur = cur + line
    if cur:
        out.append(cur)
    return "\n".join(out).strip()

def _heading(kind: str, num: str, title: str) -> str:
    title = _clean_line(title)
    return f"{num}{kind} {title}".strip() if title else f"{num}{kind}"

def _collect_heading_positions(text: str) -> Dict[str, List[Tuple[int, str]]]:
    positions: Dict[str, List[Tuple[int, str]]] = {
        "part": [],
        "subpart": [],
        "chapter": [],
        "section": [],
    }
    for m in PART_RE.finditer(text):
        positions["part"].append((m.start(), _heading("编", m.group("num"), m.group("title"))))
    for m in SUBPART_RE.finditer(text):
        positions["subpart"].append((m.start(), _heading("分编", m.group("num"), m.group("title"))))
    for m in CHAPTER_RE.finditer(text):
        positions["chapter"].append((m.start(), _heading("章", m.group("num"), m.group("title"))))
    for m in SECTION_RE.finditer(text):
        positions["section"].append((m.start(), _heading("节", m.group("num"), m.group("title"))))
    return positions

def _last_heading_before(items: List[Tuple[int, str]], pos: int) -> str:
    last = ""
    for p, val in items:
        if p <= pos:
            last = val
        else:
            break
    return last

def _strip_heading_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        line = _clean_line(raw)
        if not line:
            out.append(raw)
            continue
        if PART_RE.match(line) or SUBPART_RE.match(line) or CHAPTER_RE.match(line) or SECTION_RE.match(line):
            continue
        inline = INLINE_HEADING_RE.search(line)
        if inline:
            prefix = line[:inline.start()].strip()
            if prefix:
                out.append(prefix)
            continue
        out.append(line)
    return out

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
    lang: str
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
    text = _merge_lines(st.cur_lines)
    if not text:
        return

    article_key = st.cur_key or st.cur_no
    rec_id = f"{st.source}::{article_key}"

    out.append(
        {
            "id": rec_id,
            "law_name": st.law_name,
            "lang": st.lang,
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

def _finalize_en(st: State, out: List[Dict]) -> None:
    if not st.cur_no:
        return
    text = _merge_lines(st.cur_lines)
    if not text:
        return

    article_key = st.cur_key or st.cur_no
    rec_id = f"{st.source}::{article_key}"

    out.append(
        {
            "id": rec_id,
            "law_name": st.law_name,
            "lang": st.lang,
            "part": st.part,
            "subpart": st.subpart,
            "chapter": st.chapter,
            "section": st.section,
            "article_no": st.cur_no,
            "article_key": article_key,
            "article_id": article_key,
            "text": text,
            "source": st.source,
        }
    )
    st.cur_key = None
    st.cur_no = None
    st.cur_lines = []

def parse_english_by_lines(text: str, source: str, law_name: str) -> List[Dict]:
    st = State(law_name=law_name, source=source, lang="en")
    records: List[Dict] = []

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            if st.cur_no:
                st.cur_lines.append("")
            continue

        m = EN_ARTICLE_RE.match(line)
        if m:
            st.chapter = _heading("Article", m.group("num"), m.group("title"))
            st.section = ""
            continue

        m = EN_PART_RE.match(line)
        if m:
            st.section = _heading("Part", m.group("num"), m.group("title"))
            continue

        m = EN_SECTION_LINE_RE.match(line)
        if m:
            _finalize_en(st, records)
            key = _clean_line(m.group("id"))
            st.cur_key = key
            st.cur_no = f"§ {key}"
            st.cur_lines = [line]
            continue

        if st.cur_no:
            st.cur_lines.append(line)

    _finalize_en(st, records)
    return records

def parse_by_lines(text: str, source: str, law_name: str) -> List[Dict]:
    if detect_lang(text) == "en":
        return parse_english_by_lines(text, source=source, law_name=law_name)
    text = _normalize_article_markers(text)
    st = State(law_name=law_name, source=source, lang=detect_lang(text))
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
            if st.cur_no:
                _finalize(st, records)
            st.part = _heading("编", m.group("num"), m.group("title"))
            st.subpart = ""
            st.chapter = ""
            st.section = ""
            continue

        m = SUBPART_RE.match(line)
        if m and not ARTICLE_LINE_RE.match(line):
            if st.cur_no:
                _finalize(st, records)
            st.subpart = _heading("分编", m.group("num"), m.group("title"))
            st.chapter = ""
            st.section = ""
            continue

        m = CHAPTER_RE.match(line)
        if m and not ARTICLE_LINE_RE.match(line):
            if st.cur_no:
                _finalize(st, records)
            st.chapter = _heading("章", m.group("num"), m.group("title"))
            st.section = ""
            continue

        m = SECTION_RE.match(line)
        if m and not ARTICLE_LINE_RE.match(line):
            if st.cur_no:
                _finalize(st, records)
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
        m = ARTICLE_LINE_NO_DAI_RE.match(line)
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

def parse_english_by_scan_fallback(text: str, source: str, law_name: str) -> List[Dict]:
    matches = list(EN_SECTION_SCAN_RE.finditer(text))
    matches = sorted(matches, key=lambda m: m.start())
    if not matches:
        return []

    records: List[Dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()

        key = _clean_line(m.group("id"))
        article_no = f"§ {key}"
        rec_id = f"{source}::{key}"

        seg_text = _merge_lines(segment.splitlines())

        records.append(
            {
                "id": rec_id,
                "law_name": law_name,
                "lang": "en",
                "part": "",
                "subpart": "",
                "chapter": "",
                "section": "",
                "article_no": article_no,
                "article_key": key,
                "article_id": key,
                "text": seg_text,
                "source": source,
            }
        )
    return records

def parse_by_scan_fallback(text: str, source: str, law_name: str) -> List[Dict]:
    """
    Fallback when the input is not cleanly line-broken (common with PDF copy-paste).
    We scan the whole text for article markers and slice segments between them.
    """
    if detect_lang(text) == "en":
        return parse_english_by_scan_fallback(text, source=source, law_name=law_name)
    text = _normalize_article_markers(text)
    lang = detect_lang(text)
    heading_pos = _collect_heading_positions(text)
    matches = list(ARTICLE_SCAN_RE.finditer(text)) + list(ARTICLE_SCAN_NO_DAI_RE.finditer(text))
    matches = sorted(matches, key=lambda m: m.start())
    matches = [m for m in matches if not _is_citation_start(text, m.start())]
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

        seg_lines = _strip_heading_lines(segment.splitlines())
        seg_text = _merge_lines(seg_lines)
        part = _last_heading_before(heading_pos["part"], start)
        subpart = _last_heading_before(heading_pos["subpart"], start)
        chapter = _last_heading_before(heading_pos["chapter"], start)
        section = _last_heading_before(heading_pos["section"], start)

        records.append(
            {
                "id": rec_id,
                "law_name": law_name,
                "lang": lang,
                "part": part,
                "subpart": subpart,
                "chapter": chapter,
                "section": section,
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
    out_root = Path(cfg.paths.processed_dir)

    txt_files = sorted(raw_dir.rglob("*.txt"))
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

        lang = detect_lang(text)
        law_name = "Uniform Commercial Code" if lang == "en" else "中华人民共和国民法典"
        recs_line = parse_by_lines(text, source=p.name, law_name=law_name)
        recs_scan = parse_by_scan_fallback(text, source=p.name, law_name=law_name)
        if recs_scan and (len(recs_line) < 10 or len(recs_scan) > len(recs_line)):
            logger.warning(
                "Switching to scan fallback (line=%d scan=%d).",
                len(recs_line),
                len(recs_scan),
            )
            recs = recs_scan
        else:
            recs = recs_line

        logger.info(f"Parsed records from {p.name}: {len(recs)}")
        all_records.extend(recs)

    by_lang: Dict[str, List[Dict]] = {"zh": [], "en": []}
    for r in all_records:
        r_lang = str(r.get("lang") or "zh").strip().lower()
        if r_lang not in by_lang:
            by_lang[r_lang] = []
        by_lang[r_lang].append(r)

    total = sum(len(v) for v in by_lang.values())
    logger.info(f"Total records: {total}")
    for lang_key, recs in by_lang.items():
        if not recs:
            continue
        out_path = out_root / f"law_{lang_key}.jsonl"
        _write_jsonl(out_path, recs)
        logger.info("Saved %d records to %s", len(recs), out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
