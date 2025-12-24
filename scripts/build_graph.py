from __future__ import annotations

from legalrag.config import AppConfig
from legalrag.retrieval.builders.graph_builder import GraphBuilder
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = AppConfig.load()
    try:
        out = GraphBuilder(cfg).build_from_corpus()
        logger.info("law_graph written: %s", out)
    except Exception as e:
        logger.exception("build_graph failed: %s", e)
        raise


if __name__ == "__main__":
    main()
