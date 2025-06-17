from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

from legalrag.config import AppConfig
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    cfg = AppConfig.load()
    retriever = HybridRetriever(cfg)

    eval_path = Path(cfg.paths.eval_dir) / "contract_law_qa.jsonl"
    if not eval_path.exists():
        logger.error(f"{eval_path} 不存在，先准备评估数据。")
        return

    items = [json.loads(l) for l in eval_path.open("r", encoding="utf-8")]
    logger.info(f"Loaded {len(items)} eval items")
    stats = Counter()

    for it in items:
        q = it["question"]
        targets = set(it.get("target_articles", []))
        hits = retriever.search(q, cfg.retrieval.top_k)
        hit_articles = [h.chunk.article_no for h in hits]

        stats["total"] += 1
        if any(a in targets for a in hit_articles):
            stats["hit_at_k"] += 1

    total = stats["total"] or 1
    recall = stats["hit_at_k"] / total
    logger.info(f"Recall@{cfg.retrieval.top_k} = {recall:.3f}")
    print(f"Recall@{cfg.retrieval.top_k} = {recall:.3f} ({stats['hit_at_k']}/{total})")


if __name__ == "__main__":
    main()
