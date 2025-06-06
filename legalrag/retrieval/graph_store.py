from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from legalrag.config import AppConfig
from legalrag.models import LawNode
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class LawGraphStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.graph_path = Path(cfg.paths.law_graph_jsonl)
        self.nodes: Dict[str, LawNode] = {}

    def load(self):
        if self.nodes:
            return
        if not self.graph_path.exists():
            logger.warning(f"[Graph] {self.graph_path} 不存在，跳过 law_graph。")
            return
        with self.graph_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                node = LawNode(**obj)
                self.nodes[node.article_id] = node
        logger.info(f"[Graph] Loaded {len(self.nodes)} law nodes")

    def get_neighbors(self, article_id: str, depth: int = 1) -> List[LawNode]:
        self.load()
        if article_id not in self.nodes:
            return []
        result = []
        visited = set()
        frontier = [article_id]
        for _ in range(depth):
            next_frontier = []
            for aid in frontier:
                if aid in visited:
                    continue
                visited.add(aid)
                node = self.nodes.get(aid)
                if not node:
                    continue
                result.append(node)
                next_frontier.extend(node.neighbors)
            frontier = next_frontier
        return result
