from __future__ import annotations

from pathlib import Path

from legalrag.config import AppConfig
from legalrag.retrieval.builders.graph_builder import GraphBuilder
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    base_cfg = AppConfig.load()
    for lang in ("zh", "en"):
        cfg = base_cfg.with_lang(lang)
        corpus_path = Path(cfg.paths.law_jsonl)
        if not corpus_path.exists():
            logger.info("skip graph build: %s corpus not found: %s", lang, corpus_path)
            continue
        try:
            out = GraphBuilder(cfg).build_from_corpus()
            logger.info("law_graph written (%s): %s", lang, out)
        except Exception as e:
            logger.exception("build_graph failed (%s): %s", lang, e)
            raise


if __name__ == "__main__":
    main()
