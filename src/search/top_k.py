"""Top-k loopless routes with Yen's algorithm, on top of our 2A search methods.

OWNER: Kevin (Integration A2A)

The spec asks for up to 5 routes from O to D. One 2A search returns one path,
so we use Yen's algorithm (https://en.wikipedia.org/wiki/Yen%27s_algorithm):

  1. Route 1 = the normal 2A search from the origin.
  2. For each node on the previous route (the "spur node"):
       - keep the part of the route before it (the "root"),
       - hide the next edge of every accepted route that shares this root,
         and hide the root's nodes so the new route cannot loop back,
       - run the 2A search again from the spur node -> a detour,
       - root + detour = a candidate route.
  3. The cheapest candidate becomes the next route. Repeat until k routes.

With "AS" (A*) every inner search is optimal, so the result is the true k
fastest routes. With other methods the routes are still distinct and loopless,
just not guaranteed to be the fastest - which is itself a nice comparison for
the Research section.
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
            # hide the edge leaving the root on every accepted route with this root
            banned_edges = {(r.path[i], r.path[i + 1]) for r in accepted
                            if len(r.path) > i + 1 and r.path[: i + 1] == root}
            banned_nodes = set(root[:-1])          # no looping back into the root

            detour, created = run(spur, banned_nodes, banned_edges)
            if detour is None:
                continue
            route = tuple(root[:-1] + detour)
            if route in seen:
                continue
            seen.add(route)
            heapq.heappush(candidates, (full.path_cost(list(route)), route, created))

        if not candidates:
            break                                   # fewer than k routes exist
        cost, route, created = heapq.heappop(candidates)
        accepted.append(SearchResult(list(route), cost, created))

    return sorted(accepted, key=lambda r: (r.cost, r.path))
