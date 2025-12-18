from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from legalrag.config import AppConfig
from legalrag.schemas import LawNode
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class LawGraphStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.graph_path = Path(cfg.paths.law_graph_jsonl)
        self.nodes: Dict[str, LawNode] = {}
        self.adj: Dict[article_id, list[(neighbor_id, rel_type)]]

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

    def get_node(self, article_id: str) -> Optional[LawNode]:
        return self.nodes.get(article_id)

    def classify_relation(
        self, seed: LawNode, other: LawNode
    ) -> str:
        """
        粗略关系分类：
        - 同 chapter & 同 section & 相邻条号 → sibling
        - 同 chapter & section 不同       → sibling_section
        - chapter 不同但接近              → neighbor_chapter
        - 其它                             → neighbor
        """
        try:
            seed_no = int(seed.article_no)
            other_no = int(other.article_no)
        except Exception:
            seed_no = other_no = None

        if seed.chapter == other.chapter:
            if seed.section == other.section:
                if seed_no is not None and other_no is not None:
                    if abs(seed_no - other_no) == 1:
                        return "sibling"
                return "same_section"
            return "same_chapter"
        return "neighbor"

    def walk(
        self,
        start_ids: list[str],
        depth: int = 2,
        rel_types: list[str] | None = None,
        limit: int = 80,
    ) -> list[LawNode]:
        """
        简单 BFS：
        - 从 start_ids 出发
        - 按 rel_types 过滤边
        - 控制 depth / limit
        """
        if not start_ids:
            return []

        rel_types = rel_types or []
        visited: set[str] = set(start_ids)
        results: list[LawNode] = []
        frontier: list[tuple[str, int]] = [(sid, 0) for sid in start_ids]

        while frontier and len(results) < limit:
            current_id, d = frontier.pop(0)
            if d >= depth:
                continue

            for nb_id, rtype in self.adj.get(current_id, []):
                if nb_id in visited:
                    continue
                if rel_types and rtype not in rel_types:
                    continue

                visited.add(nb_id)
                node = self.nodes.get(nb_id)
                if not node:
                    continue

                # 可以在 node 上挂一个 graph_depth / relations 字段（pydantic 模型允许 extra=True 时）
                node.graph_depth = d + 1
                node.relations = [rtype]

                results.append(node)
                frontier.append((nb_id, d + 1))

                if len(results) >= limit:
                    break

        return results

    def get_definition_sources(self, term: str) -> list[LawNode]:
        """
        定义条款扩展：
        - 简单策略：在 nodes 中查“本法所称 term”、“是指 term”
        """
        if not term:
            return []

        term = term.strip()
        res: list[LawNode] = []
        for node in self.nodes.values():
            text = getattr(node, "text", "") or ""
            if not text:
                continue
            if term in text and any(p in text for p in ["是指", "本法所称", "本条所称", "本编所称"]):
                node.graph_depth = getattr(node, "graph_depth", 1)
                node.relations = ["definition"]
                res.append(node)
        return res