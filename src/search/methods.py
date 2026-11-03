"""Search method names (same as the 2A CLI) mapped to the 2A search functions."""
from __future__ import annotations

from typing import Callable

from src.search.a2a.algorithms import astar, bfs, cus1, cus2, dfs, gbfs

SearchFn = Callable[..., tuple]

SEARCH_METHODS: dict[str, SearchFn] = {
    "DFS": dfs.search,
    "BFS": bfs.search,
    "GBFS": gbfs.search,
    "AS": astar.search,
    "CUS1": cus1.search,
    "CUS2": cus2.search,
}


def get_search(name: str) -> SearchFn:
    try:
        return SEARCH_METHODS[name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown search method '{name}'. Available: {list(SEARCH_METHODS)}") from exc
