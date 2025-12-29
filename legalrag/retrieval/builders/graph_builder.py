from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from legalrag.config import AppConfig
from legalrag.schemas import LawChunk
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------
# Language detection 
# -----------------------
_RE_ZH = re.compile(r"[\u4e00-\u9fff]")
_RE_EN = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    """Return 'zh' or 'en' (best-effort)."""
    if not text:
        return "zh"
    zh = len(_RE_ZH.findall(text))
    en = len(_RE_EN.findall(text))
    return "en" if en > zh else "zh"


# -----------------------
# ZH parsers
# -----------------------
_RE_ZH_ARTICLE = re.compile(r"第\s*([0-9一二三四五六七八九十百千万两〇零]+)\s*条")
_RE_ZH_RANGE = re.compile(
    r"第\s*([0-9一二三四五六七八九十百千万两〇零]+)\s*条\s*(?:至|到)\s*第\s*([0-9一二三四五六七八九十百千万两〇零]+)\s*条"
)
_RE_ZH_DEFINE_STRONG = re.compile(
    r"(?:本法|本章|本节|本编|本条)?\s*所称\s*([^，。；:：\n]{1,30})\s*(?:[，,:：]\s*)?是指"
)
_RE_ZH_DEFINE_WEAK = re.compile(r"([^，。；:：\n]{2,30})\s*是指")

_TERM_STOPWORDS_ZH = {
    "本法", "本章", "本节", "本编", "本条", "当事人", "合同", "法律", "规定", "行为", "权利", "义务",
    "应当", "可以", "不得", "人民法院", "国家", "组织", "单位",
}


def _cn_to_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None

    digit = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    unit = {"十": 10, "百": 100, "千": 1000, "万": 10000}

    total = 0
    cur = 0

    for ch in s:
        if ch in digit:
            cur = digit[ch]
        elif ch in unit:
            u = unit[ch]
            if cur == 0:
                cur = 1
            if u == 10000:
                total = (total + cur) * 10000
                cur = 0
            else:
                total += cur * u
                cur = 0
        else:
            return None

    total += cur
    return total if total > 0 else None


def _make_zh_key(num: int) -> str:
    return f"第{num}条"

# -----------------------
# EN parsers
# -----------------------
# Citations: Section 2.1 / § 2-201 / Article 9 / Subsection (a)
_RE_EN_RANGE = re.compile( r"(Section|Sec\.?|§{1,2}|Article|Art\.?)\s+(\d+)\s*(?:–|-|to|through)\s*(\d+)", flags=re.IGNORECASE )
_RE_EN_SECTION = re.compile(r"(?:Section|Sec\.?)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
_RE_EN_ARTICLE = re.compile(r"(?:Article)\s+(\d+)", re.IGNORECASE)
_RE_EN_PARAGRAPH = re.compile(r"§\s*(\d+(?:-\d+)*)")
# Definitions: '"X" means ...' / 'X shall mean ...'
_RE_EN_DEFINE_QUOTED = re.compile(r"“([^”]{1,60})”\s+(?:means|shall mean)\b", re.IGNORECASE)
_RE_EN_DEFINE_QUOTED2 = re.compile(r"\"([^\"]{1,60})\"\s+(?:means|shall mean)\b", re.IGNORECASE)
_RE_EN_DEFINE_BARE = re.compile(r"\b([A-Z][A-Za-z0-9\-_ ]{1,40})\s+(?:means|shall mean)\b")
_SECTION_ID_RE = re.compile(
    r"""
    ^                      # 整个字符串
    (?P<num>\d+)           # 主数字，如 3 / 12 / 1502
    (?P<suffix>[A-Za-z])?  # 可选字母，如 3A
    (?P<trailer>           # 可选尾巴，如 (b)、(a)(1)
        (?:\([^)]+\))*
    )$
    """,
    re.VERBOSE,
)
_TERM_STOPWORDS_EN = {
    "Agreement", "Party", "Parties", "Law", "Regulation", "Court", "State", "Company",
}

_HIER_EN_RE = re.compile(
    r"""
    ^\s*
    (?:(?:Part)\s+(?P<part>[IVXLC\d]+))?\s*
    (?:(?:Chapter|Chap\.)\s+(?P<chap>[\w\d]+))?\s*
    (?:(?:Section|Sec\.?|§)\s+(?P<section>[\dA-Za-z()]+))?\s*
    (?:(?:Clause|Cl\.)\s+(?P<clause>[A-Za-z0-9()]+))?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE
)

def canonical_en_key_from_hierarchy(raw: str) -> Optional[str]:
    """
    例子：
    "Part IV Chapter 2 Section 3 Clause (b)" ->
        "en:part:4:chap:2:sec:3:clause:b"
    "Section 1502(a)(1)" ->
        "en:sec:1502(a)(1)"
    """
    m = _HIER_EN_RE.match(raw)
    if not m:
        return None

    parts = ["en"]
    part = m.group("part")
    chap = m.group("chap")
    sec = m.group("section")
    clause = m.group("clause")

    if part:
        # 罗马数字统一转成数字（你可以实现一个 roman_to_int）
        parts += ["part", part.lower()]

    if chap:
        parts += ["chap", chap.lower()]

    if sec:
        sec_id = parse_en_section_id(sec)
        if not sec_id:
            return None
        parts += ["sec", sec_id]

    if clause:
        parts += ["clause", clause.strip("()").lower()]

    return ":".join(parts)


def parse_en_section_id(raw: str) -> Optional[str]:
    """
    "3A" -> "3a"
    "12(b)" -> "12(b)"
    "1502(a)(1)" -> "1502(a)(1)"
    """
    raw = raw.strip()
    m = _SECTION_ID_RE.match(raw)
    if not m:
        return None

    num = m.group("num")
    suffix = (m.group("suffix") or "").lower()
    trailer = m.group("trailer") or ""
    return f"{num}{suffix}{trailer}"



def _safe_add(
    adj: Dict[str, List[dict]],
    src: str,
    dst: str,
    relation: str,
    conf: float,
    evidence: Optional[Dict[str, Any]],
    *,
    max_per_node: int,
) -> None:
    if not src or not dst or src == dst:
        return
    lst = adj.setdefault(src, [])
    if len(lst) >= max_per_node:
        return
    for x in lst:
        if x.get("article_id") == dst and x.get("relation") == relation:
            # keep higher confidence
            if float(x.get("conf", 0.0) or 0.0) < float(conf):
                x["conf"] = float(conf)
                if evidence:
                    x["evidence"] = evidence
            return
    edge = {"article_id": dst, "relation": relation, "conf": float(conf)}
    if evidence:
        edge["evidence"] = evidence
    lst.append(edge)


@dataclass
class GraphBuilder:
    cfg: AppConfig

    def build_from_chunks(self, chunks: List[LawChunk]) -> Path:
        out_graph = Path(self.cfg.paths.law_graph_jsonl)
        out_graph.parent.mkdir(parents=True, exist_ok=True)
        out_graph.write_text("", encoding="utf-8")

        # stable order for prev/next
        def _sort_key(c: LawChunk):
            aid = getattr(c, "article_id", "") or ""
            try:
                return (0, int(str(aid)))
            except Exception:
                return (1, str(aid))
        chunks = sorted(list(chunks), key=_sort_key)

        # config / guardrails
        rcfg = getattr(self.cfg, "retrieval", None)
        max_cite = int(getattr(rcfg, "graph_max_cite_edges_per_node", 20) if rcfg else 20)
        max_def = int(getattr(rcfg, "graph_max_defined_by_edges_per_node", 10) if rcfg else 10)
        max_total = int(getattr(rcfg, "graph_max_edges_per_node", 60) if rcfg else 60)

        # indices
        ref2id: Dict[str, str] = {}
        id2chunk: Dict[str, LawChunk] = {}

        for c in chunks:
            aid = str(getattr(c, "article_id", "") or getattr(c, "id", "") or "").strip()
            if not aid:
                continue

            id2chunk[aid] = c
            ref2id[aid] = aid

            # numeric -> "第<num>条"
            try:
                ref2id[_make_zh_key(int(aid))] = aid
            except Exception:
                pass

            # Chinese article_no: "第五百八十七条"
            ano = str(getattr(c, "article_no", "") or "").strip()
            if ano:
                ano_norm = re.sub(r"\s+", "", ano)
                if ano_norm.startswith("第") and ano_norm.endswith("条"):
                    ref2id[ano_norm] = aid
                    inner = ano_norm[1:-1]
                    n2 = _cn_to_int(inner)
                    if n2 is not None:
                        ref2id[_make_zh_key(n2)] = aid

            # English article_no: "Section 12", "Sec. 5", "Article 7"
            ano_en = str(getattr(c, "article_no_en", "") or "").strip()
            if ano_en:
                key = canonical_en_key_from_hierarchy(ano_en) 
                if key: 
                    ref2id[key] = aid

        # adjacency
        adj: Dict[str, List[dict]] = {}

        # term -> defining article id
        term2def: Dict[str, str] = {}
        def2terms: Dict[str, List[str]] = {}

        # pass 1: prev/next + cite + defines_term candidates
        for i, c in enumerate(chunks):
            aid = str(getattr(c, "article_id", "") or getattr(c, "id", "") or "").strip()
            if not aid:
                continue

            # prev/next (conf=1.0)
            if i > 0:
                _safe_add(adj, aid, str(getattr(chunks[i-1], "article_id", chunks[i-1].id)), "prev", 1.0, None, max_per_node=max_total)
            if i + 1 < len(chunks):
                _safe_add(adj, aid, str(getattr(chunks[i+1], "article_id", chunks[i+1].id)), "next", 1.0, None, max_per_node=max_total)

            text = str(getattr(c, "text", "") or "")
            if not text.strip():
                continue

            lang = detect_lang(text)

            # ---------- citations ----------
            if lang == "zh":
                # ranges （第X条至第Y条）
                for m in _RE_ZH_RANGE.finditer(text):
                    a, b = m.group(1), m.group(2)
                    na, nb = _cn_to_int(a), _cn_to_int(b)
                    if na is None or nb is None:
                        continue
                    lo, hi = (na, nb) if na <= nb else (nb, na)
                    if hi - lo > 200:
                        continue
                    for num in range(lo, hi + 1):
                        dst = ref2id.get(_make_zh_key(num))
                        if dst:
                            _safe_add(
                                adj, aid, dst, "cite", 0.95,
                                {"span": [m.start(), m.end()], "text": m.group(0)},
                                max_per_node=max_cite,
                            )
                            _safe_add(
                                adj, aid, dst, "cited", 0.95,
                                {"span": [m.start(), m.end()], "text": m.group(0)},
                                max_per_node=max_cite,
                            )
                # singles （第X条）
                for m in _RE_ZH_ARTICLE.finditer(text):
                    raw = m.group(1)
                    n = _cn_to_int(raw)
                    if n is None:
                        continue
                    dst = ref2id.get(_make_zh_key(n))
                    if dst:
                        _safe_add(
                            adj, aid, dst, "cite", 0.90,
                            {"span": [m.start(), m.end()], "text": m.group(0)},
                            max_per_node=max_cite,
                        )
                        _safe_add(
                            adj, dst, aid, "cited", 0.90,
                            {"span": [m.start(), m.end()], "text": m.group(0)},
                            max_per_node=max_cite,
                        )
            else:
                # EN: ranges (e.g.: Section 10–15, Sec. 3 to 8, §§ 20-25,  Article 5 through 9)
                for m in _RE_EN_RANGE.finditer(text):
                    start = int(m.group(2))
                    end = int(m.group(3))

                    lo, hi = (start, end) if start <= end else (end, start)
                    if hi - lo > 200:
                        continue

                    for num in range(lo, hi + 1):
                        dst = ref2id.get(f"section_{num}")  # 你需要定义英文条文 key
                        if dst:
                            meta = {"span": [m.start(), m.end()], "text": m.group(0)}

                            # A → B
                            _safe_add(adj, aid, dst, "cite", 0.85, meta, max_per_node=max_cite)

                            # B ← A
                            _safe_add(adj, dst, aid, "cited", 0.85, meta, max_per_node=max_cite)

                # EN: Section / Article / § patterns
                for m in _RE_EN_SECTION.finditer(text):
                    key = m.group(1)
                    # map "2.1" -> try "2" then "2.1" as id
                    dst = ref2id.get(key) or ref2id.get(key.split(".")[0])
                    if dst:
                        _safe_add(adj, aid, dst, "cite", 0.85, {"span": [m.start(), m.end()], "text": m.group(0)}, max_per_node=max_cite)
                        _safe_add(adj, dst, aid, "cited", 0.85, {"span": [m.start(), m.end()], "text": m.group(0)}, max_per_node=max_cite)
                for m in _RE_EN_ARTICLE.finditer(text):
                    key = m.group(1)
                    dst = ref2id.get(key)
                    if dst:
                        _safe_add(adj, aid, dst, "cite", 0.85, {"span": [m.start(), m.end()], "text": m.group(0)}, max_per_node=max_cite)
                        _safe_add(adj, dst, aid, "cited", 0.85, {"span": [m.start(), m.end()], "text": m.group(0)}, max_per_node=max_cite)
                for m in _RE_EN_PARAGRAPH.finditer(text):
                    key = m.group(1)
                    dst = ref2id.get(key) or ref2id.get(key.split("-")[0])
                    if dst:
                        _safe_add(adj, aid, dst, "cite", 0.85, {"span": [m.start(), m.end()], "text": m.group(0)}, max_per_node=max_cite)
                        _safe_add(adj, dst, aid, "cited", 0.85, {"span": [m.start(), m.end()], "text": m.group(0)}, max_per_node=max_cite)

            # ---------- definitions ----------
            defs: List[Tuple[str, float]] = []
            if lang == "zh":
                for m in _RE_ZH_DEFINE_STRONG.finditer(text):
                    term = re.sub(r"\s+", "", (m.group(1) or "").strip())
                    if not term or len(term) < 2 or len(term) > 20:
                        continue
                    if term in _TERM_STOPWORDS_ZH:
                        continue
                    defs.append((term, 0.95))
                # weak candidates (lower conf; kept in meta only by default)
                for m in _RE_ZH_DEFINE_WEAK.finditer(text):
                    term = re.sub(r"\s+", "", (m.group(1) or "").strip())
                    if not term or len(term) < 2 or len(term) > 12:
                        continue
                    if term in _TERM_STOPWORDS_ZH:
                        continue
                    # do not register weak as definition edge directly; store as meta candidate
                    defs.append((term, 0.60))
            else:
                for m in _RE_EN_DEFINE_QUOTED.finditer(text):
                    term = (m.group(1) or "").strip()
                    if not term or len(term) < 2 or len(term) > 50 or term in _TERM_STOPWORDS_EN:
                        continue
                    defs.append((term, 0.95))
                for m in _RE_EN_DEFINE_QUOTED2.finditer(text):
                    term = (m.group(1) or "").strip()
                    if not term or len(term) < 2 or len(term) > 50 or term in _TERM_STOPWORDS_EN:
                        continue
                    defs.append((term, 0.95))
                for m in _RE_EN_DEFINE_BARE.finditer(text):
                    term = (m.group(1) or "").strip()
                    if not term or len(term) < 2 or len(term) > 40 or term in _TERM_STOPWORDS_EN:
                        continue
                    defs.append((term, 0.70))

            if defs:
                # keep unique by best conf
                best: Dict[str, float] = {}
                for t, cf in defs:
                    best[t] = max(best.get(t, 0.0), float(cf))
                # store candidates in node meta later
                def2terms[aid] = sorted(best.keys(), key=len, reverse=True)
                # register only strong definitions (>=0.8) into term2def
                for t, cf in best.items():
                    if cf >= 0.8 and t not in term2def:
                        term2def[t] = aid

        # pass 2: defined_by / defines_term edges (budgeted)
        if term2def:
            terms = sorted(term2def.keys(), key=len, reverse=True)
            for c in chunks:
                aid = str(getattr(c, "article_id", "") or getattr(c, "id", "") or "").strip()
                if not aid:
                    continue
                text = str(getattr(c, "text", "") or "")
                if not text.strip():
                    continue
                # avoid self edges; add top-k matches
                added = 0
                for term in terms:
                    def_id = term2def.get(term)
                    if not def_id or def_id == aid:
                        continue
                    if term in text:
                        # confidence heuristic: longer terms are usually safer
                        conf = 0.90 if len(term) >= 4 else 0.85
                        _safe_add(adj, aid, def_id, "defined_by", conf, {"term": term}, max_per_node=max_def)
                        _safe_add(adj, def_id, aid, "defines_term", conf, {"term": term}, max_per_node=max_def)
                        added += 1
                        if added >= max_def:
                            break

        # write nodes
        with out_graph.open("a", encoding="utf-8") as f:
            for c in chunks:
                aid = str(getattr(c, "article_id", "") or getattr(c, "id", "") or "").strip()
                if not aid:
                    continue
                node = {
                    "article_id": aid,
                    "article_no": getattr(c, "article_no", None),
                    "law_name": getattr(c, "law_name", None),
                    "title": getattr(c, "title", None),
                    "chapter": getattr(c, "chapter", None),
                    "section": getattr(c, "section", None),
                    "neighbors": adj.get(aid, []),
                    "meta": {
                        "defines_terms": def2terms.get(aid, []),
                    },
                }
                f.write(json.dumps(node, ensure_ascii=False) + "\n")

        logger.info(
            "[GRAPH] built %d nodes -> %s (relations include prev/next/cite/defined_by/defines_term)",
            len(chunks),
            out_graph,
        )
        return out_graph

    def build_from_corpus(self) -> Path:
        corpus = Path(self.cfg.paths.law_jsonl)
        if not corpus.exists():
            raise FileNotFoundError(f"{corpus} not found; run preprocess first.")
        chunks = [json.loads(l) for l in corpus.open("r", encoding="utf-8") if l.strip()]
        # convert to LawChunk for consistent fields
        law_chunks: List[LawChunk] = []
        for obj in chunks:
            law_chunks.append(LawChunk.model_validate(obj) if hasattr(LawChunk, "model_validate") else LawChunk(**obj))
        return self.build_from_chunks(law_chunks)
