from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Set

from legalrag.schemas import LawChunk

def iter_chunks_from_dir(processed_dir: str, pattern: str = "*.jsonl") -> Iterable[LawChunk]:
    pdir = Path(processed_dir)
    files = sorted(pdir.glob(pattern))
    for fp in files:
        if fp.is_dir():
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield LawChunk(**json.loads(line))

def load_chunks_from_dir(processed_dir: str, pattern: str = "*.jsonl") -> List[LawChunk]:
    p = Path(processed_dir)
    files = sorted(p.glob(pattern))
    seen: Set[str] = set()
    out: List[LawChunk] = []
    for fp in files:
        if not fp.is_file():
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = LawChunk(**json.loads(line))
                if c.id in seen:
                    continue
                seen.add(c.id)
                out.append(c)
    return out
