"""Shared graph data structure for the Boroondara road network.

This file is SHARED: Kevin builds the graph, Kevin searches it.
Only change it after agreeing in the group chat / a PR review.

Nodes  = SCATS sites (intersections), keyed by SCATS number (int).
Edges  = directed road links A -> B with a distance in km. The *cost* of an edge
         (travel time) is not stored here; it is computed at query time by
         src/routing/travel_time.py because it depends on the predicted flow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Node:
    scats: int
    lat: float
    lon: float
    name: str = ""                      # e.g. "WARRIGAL_RD/TOORAK_RD"
    locations: list[str] = field(default_factory=list)   # approach names in the dataset


@dataclass
class Edge:
    src: int
    dst: int
    distance_km: float
    road: str = ""


class RoadGraph:
    """A small directed graph with adjacency lists."""

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self._adj: dict[int, list[Edge]] = {}

    # ---- building -------------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.nodes[node.scats] = node
        self._adj.setdefault(node.scats, [])

    def add_edge(self, src: int, dst: int, distance_km: float | None = None, road: str = "") -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(f"Both nodes must exist before adding edge {src}->{dst}")
        if distance_km is None:
            distance_km = self.haversine_km(src, dst)
        self._adj[src].append(Edge(src, dst, distance_km, road))

    # ---- querying -------------------------------------------------------
    def neighbours(self, scats: int) -> list[Edge]:
        return list(self._adj.get(scats, []))

    def edges(self) -> list[Edge]:
        return [e for edges in self._adj.values() for e in edges]

    def has_node(self, scats: int) -> bool:
        return scats in self.nodes

    def haversine_km(self, a: int, b: int) -> float:
        """Great-circle distance between two sites (also handy as an A* heuristic)."""
        n1, n2 = self.nodes[a], self.nodes[b]
        r = 6371.0
        p1, p2 = math.radians(n1.lat), math.radians(n2.lat)
        dp = p2 - p1
        dl = math.radians(n2.lon - n1.lon)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(h))

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"RoadGraph(nodes={len(self.nodes)}, edges={len(self.edges())})"
