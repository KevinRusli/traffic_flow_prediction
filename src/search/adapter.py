"""TravelTimeProblem: the Assignment 2A `problem` contract on the Boroondara graph.

OWNER: Kevin (Integration A2A)

Our 2A algorithms (src/search/a2a/algorithms/*.py) only talk to a `problem`
object. This class gives them exactly the same interface as 2A's problem.py,
but the nodes are SCATS sites and the edge cost is the predicted travel time
in seconds instead of a cost read from a test file:

    problem.origin              origin SCATS site (int)
    problem.destinations        list of destination sites
    problem.graph               {site: [(neighbour, cost), ...]}  (used by CUS1)
    problem.neighbors(n)        problem.graph[n], sorted ascending
    problem.h(n)                straight-line time to the nearest destination
    problem.is_goal(n)          True if n is a destination
    problem.path_cost(path)     total seconds along a path, None if invalid

Heuristic h(n): straight-line distance / speed limit, in seconds. Every edge is
at least as long as the straight line between its ends, no car drives faster
than the speed limit, and the 30 s intersection delay only adds time, so h never
overestimates: A* (AS) still returns the fastest route.

`banned_nodes` / `banned_edges` hide parts of the graph. Yen's top-k algorithm
(top_k.py) uses them to force each search to find a different route.
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

        # Build the 2A-style adjacency dict once. Every node gets a key (CUS1
        # reverses the graph and expects all nodes to be present).
        self.graph: dict[int, list[tuple[int, float]]] = {}
        for site in road_graph.nodes:
            out = []
            if site not in banned_nodes:
                for edge in road_graph.neighbours(site):
                    if edge.dst in banned_nodes or (edge.src, edge.dst) in banned_edges:
                        continue
                    out.append((edge.dst, cost_fn(edge)))
            # 2A contract: sorted ascending (by neighbour number, then cost)
            self.graph[site] = sorted(out)

    # ---- the 2A problem contract ----------------------------------------
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
