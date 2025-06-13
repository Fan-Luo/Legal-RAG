from __future__ import annotations

import json
import re
from pathlib import Path

from legalrag.config import AppConfig
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

ARTICLE_RE = re.compile(r"^第(?P<num>\d+)条")


def parse_contract_law(text: str, source: str) -> list[dict]:
    lines = text.splitlines()
    records = []
    current_article_no = None
    current_text = []
    law_name = "中华人民共和国民法典·合同编"
    chapter = None
    section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("第") and line.endswith("章"):
            chapter = line
            continue
        if line.startswith("第") and line.endswith("节"):
            section = line
            continue
        m = ARTICLE_RE.match(line)
        if m:
            if current_article_no is not None:
                records.append(
                    {
                        "id": f"{source}-{current_article_no}",
                        "law_name": law_name,
                        "chapter": chapter,
                        "section": section,
                        "article_no": f"第{current_article_no}条",
                        "text": "\n".join(current_text),
                        "source": source,
                    }
                )
            current_article_no = m.group("num")
            current_text = [line]
        else:
            if current_article_no is not None:
                current_text.append(line)

    if current_article_no is not None and current_text:
        records.append(
            {
                "id": f"{source}-{current_article_no}",
                "law_name": law_name,
                "chapter": chapter,
                "section": section,
                "article_no": f"第{current_article_no}条",
                "text": "\n".join(current_text),
                "source": source,
            }
        )
    return records


def main():
    cfg = AppConfig.load()
    raw_dir = Path(cfg.paths.raw_dir)
    out_path = Path(cfg.paths.contract_law_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    total = 0
    for txt in raw_dir.glob("*.txt"):
        logger.info(f"Parsing law file: {txt.name}")
        text = txt.read_text(encoding="utf-8")
        recs = parse_contract_law(text, txt.name)
        with out_path.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(recs)
    logger.info(f"Total records: {total}, saved to {out_path}")


if __name__ == "__main__":
    main()
