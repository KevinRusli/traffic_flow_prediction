"""TravelTimeProblem: the Assignment 2A `problem` interface on the Boroondara graph.

OWNER: Kevin

Our 2A algorithms in src/search/a2a/ run unchanged on this class. Nodes are
SCATS sites and the edge cost is the predicted travel time in seconds.

h(n) is the straight-line distance to the nearest destination driven at the
speed limit. No edge is shorter than that line and nobody drives faster than
the limit, so h never overestimates and A* (AS) still finds the fastest route.

banned_nodes / banned_edges hide part of the graph; Yen's algorithm uses them.
"""
from __future__ import annotations

from typing import Callable, Iterable

from src.graph.graph import Edge, RoadGraph

CostFn = Callable[[Edge], float]


class TravelTimeProblem:
    def __init__(
        self,
        road_graph: RoadGraph,
        origin: int,
        destinations: Iterable[int],
        cost_fn: CostFn,
        speed_limit_kmh: float = 60.0,
        banned_nodes: Iterable[int] = (),
        banned_edges: Iterable[tuple[int, int]] = (),
    ) -> None:
        self.road_graph = road_graph
        self.origin = origin
        self.destinations = sorted(set(destinations))
        self.speed_limit_kmh = speed_limit_kmh

        banned_nodes = set(banned_nodes)
        banned_edges = set(banned_edges)

        # CUS1 reverses this dict, so every node needs a key even with no edges.
        self.graph: dict[int, list[tuple[int, float]]] = {}
        for site in road_graph.nodes:
            out = []
            if site not in banned_nodes:
                for edge in road_graph.neighbours(site):
                    if edge.dst in banned_nodes or (edge.src, edge.dst) in banned_edges:
                        continue
                    out.append((edge.dst, cost_fn(edge)))
            self.graph[site] = sorted(out)

    def neighbors(self, n: int) -> list[tuple[int, float]]:
        return self.graph.get(n, [])

    def is_goal(self, n: int) -> bool:
        return n in self.destinations

    def h(self, n: int) -> float:
        """Seconds needed to drive the straight line to the nearest destination."""
        if not self.destinations:
            return 0.0
        km = min(self.road_graph.haversine_km(n, d) for d in self.destinations)
        return km / self.speed_limit_kmh * 3600.0

    def path_cost(self, path: list[int]) -> float | None:
        """Total cost of `path`, or None if a step is not an edge of this problem."""
        total = 0.0
        for a, b in zip(path, path[1:]):
            step = next((c for nb, c in self.graph.get(a, []) if nb == b), None)
            if step is None:
                return None
            total += step
        return total
