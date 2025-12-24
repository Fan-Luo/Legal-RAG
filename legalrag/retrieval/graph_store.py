from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from legalrag.config import AppConfig
from legalrag.schemas import LawNode
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)


class LawGraphStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.graph_path = Path(cfg.paths.law_graph_jsonl)

        # nodes: article_id -> LawNode
        self.nodes: Dict[str, LawNode] = {}

        # adj: article_id -> List[(neighbor_id, relation_type)]
        self.adj: Dict[str, List[Tuple[str, str]]] = {}

        self._loaded = False

    def load(self):
        """
        Load graph nodes and build adjacency.
        Safe to call multiple times.
        """
        if self._loaded:
            return

        if not self.graph_path.exists():
            logger.warning(f"[Graph] {self.graph_path} 不存在，跳过 law_graph。")
            self._loaded = True
            return

        with self.graph_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                node = LawNode(**obj)
                self.nodes[node.article_id] = node

        logger.info(f"[Graph] Loaded {len(self.nodes)} law nodes")

        self._build_adj()
        self._loaded = True

    def _build_adj(self):
        """
        Build adjacency list from node.neighbors.
        node.neighbors 被认为是 article_id 列表。
        """
        adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for node in self.nodes.values():
            src_id = node.article_id
            neighbors = getattr(node, "neighbors", []) or []

            for nb_id in neighbors:
                if nb_id not in self.nodes:
                    continue

                nb_node = self.nodes[nb_id]
                rtype = self.classify_relation(node, nb_node)

                adj[src_id].append((nb_id, rtype))
                # 默认认为是双向关系
                adj[nb_id].append((src_id, rtype))

        self.adj = dict(adj)

        logger.info(
            f"[Graph] Built adjacency: {sum(len(v) for v in self.adj.values())} edges"
        )

    def get_neighbors(self, article_id: str, depth: int = 1) -> List[LawNode]:
        self.load()

        if article_id not in self.nodes:
            return []

        result: List[LawNode] = []
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
                next_frontier.extend(getattr(node, "neighbors", []) or [])

            frontier = next_frontier

        return result

    def get_node(self, article_id: str) -> Optional[LawNode]:
        self.load()
        return self.nodes.get(article_id)

    def classify_relation(self, seed: LawNode, other: LawNode) -> str:
        """
        粗略关系分类：
        - 同 chapter & 同 section & 相邻条号 → sibling
        - 同 chapter & 同 section            → same_section
        - 同 chapter 不同 section            → same_chapter
        - 其它                               → neighbor
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
        start_ids: List[str],
        depth: int = 2,
        rel_types: Optional[List[str]] = None,
        limit: int = 80,
    ) -> List[LawNode]:
        """
        BFS over graph adjacency.
        Gracefully degrades if graph is empty.
        """
        self.load()

        if not start_ids or not self.adj:
            return []

        rel_types = rel_types or []

        visited: set[str] = set(start_ids)
        results: List[LawNode] = []
        frontier: List[Tuple[str, int]] = [(sid, 0) for sid in start_ids]

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

                node.graph_depth = d + 1
                node.relations = [rtype]

                results.append(node)
                frontier.append((nb_id, d + 1))

                if len(results) >= limit:
                    break

        return results

    def get_definition_sources(self, term: str) -> List[LawNode]:
        """
        定义条款扩展
        """
        self.load()

        if not term:
            return []

        term = term.strip()
        res: List[LawNode] = []

        for node in self.nodes.values():
            text = getattr(node, "text", "") or ""
            if not text:
                continue

            if term in text and any(
                p in text for p in ["是指", "本法所称", "本条所称", "本编所称"]
            ):
                node.graph_depth = getattr(node, "graph_depth", 1)
                node.relations = ["definition"]
                res.append(node)

        return res
