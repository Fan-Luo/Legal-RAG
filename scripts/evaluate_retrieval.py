#!/usr/bin/env python3
"""
Improved retrieval evaluation script for LegalRAG.

Usage examples:
  python evaluate_retrieval.py --help
  python evaluate_retrieval.py --eval-path data/eval/law_qa.jsonl --top-k 15
  python evaluate_retrieval.py --systems bm25,dense,fused --output results.csv
  python evaluate_retrieval.py --limit 50 --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set, Dict, Any

import pandas as pd
from tqdm import tqdm

from legalrag.config import AppConfig
from legalrag.retrieval.hybrid_retriever import HybridRetriever
from legalrag.routing.router import QueryRouter
from legalrag.llm.client import LLMClient
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)

def hit_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    return float(any(h.strip() in gold for h in pred[:k]))


def recall_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(pred[:k]) & gold) / len(gold)


def mrr_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    for i, x in enumerate(pred[:k], 1):
        if x in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    def dcg(xs: List[str]) -> float:
        return sum((1.0 if x in gold else 0.0) / math.log2(i + 1) for i, x in enumerate(xs[:k], 1))

    ideal = dcg(list(gold))
    if ideal <= 1e-12:
        return 0.0
    return dcg(pred) / ideal


def get_hit_ids(hits: List[Any]) -> List[str]:
    """Extract unique article_ids from retrieval hits."""
    return list(dict.fromkeys(
        str(getattr(h.chunk, "article_id", "") or "")
        for h in hits if getattr(h.chunk, "article_id", "")
    ))


def evaluate_one(
    query: str,
    positives: List[str],
    retriever: HybridRetriever,
    router: QueryRouter,
    top_k: int,
    seed_k: int | None = None,
    verbose: bool = False
) -> Dict[str, Any]:
    gold = set(map(str.strip, positives))

    decision = router.route(query)

    eff_top_k = top_k * 8  # generous fetch for fusion
    seed_k = seed_k or max(10, top_k * 3)

    # Retrieve from each system
    dense_hits = retriever.search_dense(query, eff_top_k)
    bm25_hits = retriever.search_bm25(query, eff_top_k)
    colbert_hits = retriever.search_colbert(query, eff_top_k)

    # Fusion
    fused_hits = retriever._fuse(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        colbert_hits=colbert_hits
    )

    # Graph augmentation
    seeds = fused_hits[:seed_k]
    graph_hits = retriever.search_graph(query, eff_top_k, decision=decision, seeds=seeds)
    fused_with_graph = seeds + graph_hits

    # Full retrieval (end-to-end)
    hybrid_hits = retriever.search(query, top_k=eff_top_k, decision=decision)

    systems = {
        "bm25": get_hit_ids(bm25_hits),
        "dense": get_hit_ids(dense_hits),
        "colbert": get_hit_ids(colbert_hits),
        "fused": get_hit_ids(fused_hits),
        "fused+graph": get_hit_ids(fused_with_graph),
        "hybrid": get_hit_ids(hybrid_hits),
    }

    metrics = {}
    for name, pred in systems.items():
        metrics[name] = {
            "R@5": recall_at_k(pred, gold, 5),
            "R@10": recall_at_k(pred, gold, 10),
            "MRR@10": mrr_at_k(pred, gold, 10),
            "nDCG@10": ndcg_at_k(pred, gold, 10),
            "Hit@3": hit_at_k(pred, gold, 3),
            "Hit@10": hit_at_k(pred, gold, 10),
        }

        if verbose:
            logger.info(f"Query: {query}")
            logger.info(f"{name} → R@5: {metrics[name]['R@5']:.3f} | MRR@10: {metrics[name]['MRR@10']:.3f}")

    return {"query": query, "gold": list(gold), "systems": systems, "metrics": metrics}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance for LegalRAG systems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--eval-path",
        type=Path,
        default=Path("data/eval/law_qa.jsonl"),
        help="Path to evaluation JSONL file (each line: {'query': str, 'article_id': str})"
    )

    parser.add_argument(
        "--systems",
        type=str,
        default="bm25,dense,colbert,fused,hybrid",
        help="Comma-separated list of systems to evaluate (bm25, dense, colbert, fused, hybrid)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Main top-k for final metrics (R@K, MRR@K, nDCG@K)"
    )

    parser.add_argument(
        "--seed-k",
        type=int,
        default=None,
        help="Number of fused seeds for graph augmentation (default: 3×top-k)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (useful for quick tests)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save detailed results as CSV/JSON (e.g. results.csv)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-query results"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional custom AppConfig YAML/JSON path"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = AppConfig.load(None)
    eval_path = args.eval_path

    if not eval_path.exists():
        logger.error(f"Evaluation file not found: {eval_path}")
        logger.info("Run scripts/generate_synthetic_data.py first")
        return

    items = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[:args.limit]
        logger.info(f"Limited evaluation to first {args.limit} queries")

    logger.info(f"Loaded {len(items)} evaluation queries")

    retriever = HybridRetriever(cfg)
    llm_client = LLMClient.from_config(cfg)
    router = QueryRouter(llm_client=llm_client, llm_based=cfg.routing.llm_based)

    desired_systems = {s.strip() for s in args.systems.split(",")}

    results = []
    for item in tqdm(items, desc="Evaluating queries"):
        query = item["query"]
        positives = [item["article_id"]]  # assuming single positive for now

        try:
            res = evaluate_one(
                query=query,
                positives=positives,
                retriever=retriever,
                router=router,
                top_k=args.top_k,
                seed_k=args.seed_k,
                verbose=args.verbose
            )
            results.append(res)
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")

    if not results:
        logger.warning("No successful evaluations")
        return

    # Flatten metrics for summary
    flat_rows = []
    for r in results:
        for sys_name, ms in r["metrics"].items():
            if sys_name not in desired_systems:
                continue
            flat_rows.append({
                "query": r["query"],
                "system": sys_name,
                **ms
            })

    df = pd.DataFrame(flat_rows)

    # Summary statistics
    summary = df.groupby("system")[["R@5", "R@10", "MRR@10", "nDCG@10", "Hit@3", "Hit@10"]].agg(
        ["mean", "std", "count"]
    ).round(3)

    print("\nEvaluation Summary (mean ± std):")
    print(summary)

    # Save detailed results  
    if args.output:
        fmt = args.output.suffix.lower()
        if fmt == ".csv":
            df.to_csv(args.output, index=False)
            logger.info(f"Detailed results saved to {args.output}")
        elif fmt == ".json":
            df.to_json(args.output, orient="records", lines=True, force_ascii=False)
            logger.info(f"Detailed results saved to {args.output}")
        else:
            logger.warning(f"Unknown format {fmt}, saving as CSV")
            df.to_csv(args.output.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()