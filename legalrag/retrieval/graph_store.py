from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from legalrag.schemas import LawNode, Neighbor
from legalrag.config import AppConfig
from legalrag.utils.logger import get_logger

logger = get_logger(__name__)



class LawGraphStore:
    """
    Loads a law graph JSONL and supports BFS-style walk.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.graph_path = Path(getattr(cfg.paths, "law_graph_jsonl"))
        self.nodes: Dict[str, LawNode] = {}
        self.adj: Dict[str, List[Tuple[str, str, float, Optional[Dict[str, Any]]]]] = defaultdict(list)
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph JSONL not found: {self.graph_path}")

        nodes: Dict[str, LawNode] = {}
        with self.graph_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                aid = str(obj.get("article_id") or obj.get("id") or "").strip()
                if not aid:
                    continue
                neighbors_raw = obj.get("neighbors") or []
                neighbors: List[Neighbor] = []
                for nb in neighbors_raw:
                    if isinstance(nb, str):
                        neighbors.append(Neighbor(article_id=str(nb), relation="neighbor", conf=1.0))
                    elif isinstance(nb, dict):
                        dst = str(nb.get("article_id") or nb.get("id") or "").strip()
                        if not dst:
                            continue
                        neighbors.append(
                            Neighbor(
                                article_id=dst,
                                relation=str(nb.get("relation") or "neighbor"),
                                conf=float(nb.get("conf", 1.0) or 1.0),
                                evidence=nb.get("evidence"),
                            )
                        )
                node = LawNode(
                    article_id=aid,
                    article_no=str(obj.get("article_no") or ""),
                    law_name=obj.get("law_name"),
                    title=obj.get("title"),
                    chapter=obj.get("chapter"),
                    section=obj.get("section"),
                    neighbors=neighbors,
                    meta=obj.get("meta") or {},
                )
                nodes[aid] = node

        self.nodes = nodes

        # Build adjacency  
        adj: Dict[str, List[Tuple[str, str, float, Optional[Dict[str, Any]]]]] = defaultdict(list)
        edge_cnt = 0
        for src, node in self.nodes.items():
            for e in node.neighbors:
                adj[src].append((e.article_id, e.relation, float(e.conf or 1.0), e.evidence))
                edge_cnt += 1
        self.adj = adj

        self._loaded = True
        logger.info("[Graph] Loaded %d law nodes", len(self.nodes))
        logger.info("[Graph] Built adjacency: %d edges", edge_cnt)

    def walk(
        self,
        start_ids: List[str],
        limit: int = 80,
        relation_max_depth: Optional[Dict[str, int]] = None,
        rel_types: Optional[List[str]] = None,
        min_conf: float = 0.0,
    ) -> List[LawNode]:
        """
        BFS from start_ids up to `depth` hops, returning at most `limit` unique nodes.
        - rel_types: if provided and non-empty, only traverse these relation names
        - min_conf: discard edges with conf < min_conf
        Returned nodes are *cloned* with query-time fields populated, avoiding global state pollution.
        """
        self.load()

        start_ids = [str(x).strip() for x in (start_ids or []) if str(x).strip()]
        if not start_ids:
            return []

        rcfg = self.cfg.retrieval
        if relation_max_depth is None:
            relation_max_depth = rcfg.graph_walk_depths if hasattr(rcfg, "graph_walk_depths") else {"default": 2}
        if rel_types is None:
            rel_types = getattr(rcfg, "graph_rel_types", None) if rcfg else None

        default_max_d = relation_max_depth.get("default", 2)
        limit = max(1, int(limit))
        rel_allow = set([str(r) for r in rel_types]) if rel_types else None
        min_conf = float(min_conf or 0.0)

        visited: set[str] = set(start_ids)
        # queue items: (node_id, dist, parent_id, relation_used)
        q: deque[Tuple[str, int, Optional[str], Optional[str]]] = deque()
        for sid in start_ids:
            q.append((sid, 0, None, None))

        results: List[LawNode] = []
        while q and len(results) < limit:
            cur, dist, parent, rel = q.popleft()

            if rel:
                max_allowed = relation_max_depth.get(rel, default_max_d)
            else:
                max_allowed = default_max_d
                
     
            if dist >= max_allowed:
                continue

            for nb_id, rtype, conf, evidence in self.adj.get(cur, []):
                if min_conf > 0 and conf < min_conf:
                    continue
                if rel_allow is not None and rtype not in rel_allow:
                    continue
                if nb_id in visited:
                    continue
                visited.add(nb_id)

                n0 = self.nodes.get(nb_id)
                if not n0:
                    continue

                n = copy.copy(n0)
                # query-time fields
                n.graph_depth = dist + 1
                n.graph_parent = cur
                n.relations = rtype
                n.relations = [rtype]
                # attach evidence for debugging (non-schema; safe in meta)
                if evidence:
                    n.meta = dict(n.meta or {})
                    n.meta["_edge_evidence"] = evidence
                    n.meta["_edge_conf"] = conf

                results.append(n)
                if len(results) >= limit:
                    break
                q.append((nb_id, dist + 1, cur, rtype))

        return results

    def get_neighbors(self, article_id: str, depth: int = 1) -> List[LawNode]:
        self.load()
        aid0 = str(article_id).strip()
        if aid0 not in self.nodes:
            return []

        depth = max(1, int(depth))
        visited = {aid0}
        frontier = [aid0]
        out: List[LawNode] = []

        for _ in range(depth):
            next_frontier: List[str] = []
            for aid in frontier:
                for nb_id, rtype, conf, evidence in self.adj.get(aid, []):
                    if nb_id in visited:
                        continue
                    visited.add(nb_id)
                    n0 = self.nodes.get(nb_id)
                    if not n0:
                        continue
                    out.append(n0)
                    next_frontier.append(nb_id)
            frontier = next_frontier

        return out

    def get_node(self, article_id: str) -> Optional[LawNode]:
        return self.nodes.get(article_id)