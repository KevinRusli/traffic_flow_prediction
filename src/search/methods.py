"""Name -> search function, using the same method names as the 2A CLI.

    from src.search.methods import SEARCH_METHODS
    goal, nodes_created, path = SEARCH_METHODS["AS"](problem)
"""
from __future__ import annotations

from typing import Callable

from src.search.a2a.algorithms import astar, bfs, cus1, cus2, dfs, gbfs

SearchFn = Callable[..., tuple]

SEARCH_METHODS: dict[str, SearchFn] = {
    "DFS": dfs.search,
    "BFS": bfs.search,
    "GBFS": gbfs.search,
    "AS": astar.search,
    "CUS1": cus1.search,   # bidirectional BFS (fewest edges)
    "CUS2": cus2.search,   # A* on number of moves (fewest edges)
}

#: Methods guaranteed to return the cheapest (= fastest) path.
OPTIMAL_METHODS = {"AS"}


def get_search(name: str) -> SearchFn:
    try:
        return SEARCH_METHODS[name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown search method '{name}'. Available: {list(SEARCH_METHODS)}") from exc
