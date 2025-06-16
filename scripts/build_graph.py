from __future__ import annotations

import json
from pathlib import Path

from legalrag.config import AppConfig
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    cfg = AppConfig.load()
    processed = Path(cfg.paths.contract_law_jsonl)
    out_graph = Path(cfg.paths.law_graph_jsonl)

    if not processed.exists():
        logger.error(f"{processed} 不存在，请先运行 preprocess_law.")
        return

    out_graph.parent.mkdir(parents=True, exist_ok=True)
    out_graph.unlink(missing_ok=True)

    # 简单 graph：每条法条一个节点，相邻条文互为 neighbor
    chunks = [json.loads(l) for l in processed.open("r", encoding="utf-8")]
    for i, c in enumerate(chunks):
        neighbors = []
        if i > 0:
            neighbors.append(chunks[i-1]["id"])
        if i < len(chunks) - 1:
            neighbors.append(chunks[i+1]["id"])
        node = {
            "article_id": c["id"],
            "article_no": c["article_no"],
            "chapter": c.get("chapter"),
            "section": c.get("section"),
            "neighbors": neighbors,
            "meta": {},
        }
        out_graph.write_text("", encoding="utf-8") if i == 0 else None
        with out_graph.open("a", encoding="utf-8") as f:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")

    logger.info(f"law_graph 写入 {out_graph}")


if __name__ == "__main__":
    main()
