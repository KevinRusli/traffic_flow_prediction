"""Up to k different routes with Yen's algorithm, using our 2A search methods.

OWNER: Kevin

1. Route 1 is a normal 2A search from the origin.
2. For each node of the last accepted route (the spur node), hide the edges
   that earlier routes took from the same prefix, hide the prefix's nodes, and
   search again from the spur node. Prefix + new path is a candidate.
3. The cheapest candidate becomes the next route. Repeat until k routes.

With AS every inner search is optimal, so the result is the k fastest routes.
Other methods still give distinct loopless routes, but not necessarily the fastest.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

from src.graph.graph import RoadGraph
from src.search.adapter import CostFn, TravelTimeProblem
from src.search.methods import get_search


@dataclass
class SearchResult:
    path: list[int]          # SCATS sites from origin to destination
    cost: float              # total travel time in seconds
    nodes_created: int = 0   # summed over every 2A search Yen ran for this route


def k_shortest_paths(
    graph: RoadGraph,
    origin: int,
    goal: int,
    cost_fn: CostFn,
    k: int = 5,
    method: str = "AS",
    speed_limit_kmh: float = 60.0,
) -> list[SearchResult]:
    """Return up to k distinct loopless routes, sorted by travel time."""
    search = get_search(method)

    def run(start: int, banned_nodes=(), banned_edges=()):
        problem = TravelTimeProblem(graph, start, [goal], cost_fn, speed_limit_kmh,
                                    banned_nodes, banned_edges)
        found, created, path = search(problem)
        return (path if found is not None else None), created

    full = TravelTimeProblem(graph, origin, [goal], cost_fn, speed_limit_kmh)

    first, created = run(origin)
    if first is None:
        return []
    accepted = [SearchResult(first, full.path_cost(first), created)]
    candidates: list[tuple[float, tuple[int, ...], int]] = []   # (cost, path, nodes_created)
    seen = {tuple(first)}

    while len(accepted) < k:
        prev = accepted[-1].path
        for i in range(len(prev) - 1):
            spur, root = prev[i], prev[: i + 1]
            banned_edges = {(r.path[i], r.path[i + 1]) for r in accepted
                            if len(r.path) > i + 1 and r.path[: i + 1] == root}
            banned_nodes = set(root[:-1])

            detour, created = run(spur, banned_nodes, banned_edges)
            if detour is None:
                continue
            route = tuple(root[:-1] + detour)
            if route in seen:
                continue
            seen.add(route)
            heapq.heappush(candidates, (full.path_cost(list(route)), route, created))

        if not candidates:
            break
        cost, route, created = heapq.heappop(candidates)
        accepted.append(SearchResult(list(route), cost, created))

    return sorted(accepted, key=lambda r: (r.cost, r.path))
