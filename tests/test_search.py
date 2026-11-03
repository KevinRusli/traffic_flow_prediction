"""2A search integration + top-k tests - OWNER: Kevin.  (TC17-TC20, TC26, TC28)

These run our UNCHANGED 2A algorithms through TravelTimeProblem on toy graphs,
with edge cost = distance so the expected answers are easy to check by hand.
"""
import pytest

from src.search.adapter import TravelTimeProblem
from src.search.methods import SEARCH_METHODS
from src.search.top_k import k_shortest_paths


def dist_cost(edge):
    return edge.distance_km


def problem(graph, origin, goal, **kw):
    return TravelTimeProblem(graph, origin, [goal], dist_cost, **kw)


def _valid_path(graph, path, origin, goal):
    assert path[0] == origin and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert b in {e.dst for e in graph.neighbours(a)}, f"no edge {a}->{b}"


def _all_simple_paths(graph, a, b, path=None):
    path = path or [a]
    if path[-1] == b:
        yield list(path)
        return
    for e in graph.neighbours(path[-1]):
        if e.dst not in path:
            yield from _all_simple_paths(graph, a, b, path + [e.dst])


def test_tc17_astar_optimal(toy_graph):
    goal, created, path = SEARCH_METHODS["AS"](problem(toy_graph, 1, 4))
    assert goal == 4 and path == [1, 2, 4]
    assert problem(toy_graph, 1, 4).path_cost(path) == pytest.approx(2.0)


@pytest.mark.parametrize("method", list(SEARCH_METHODS))
def test_tc18_every_2a_method_returns_valid_path(method, toy_graph):
    goal, created, path = SEARCH_METHODS[method](problem(toy_graph, 1, 4))
    assert goal == 4 and created >= 1
    _valid_path(toy_graph, path, 1, 4)


def test_tc19_top_k_sorted_distinct_loopless(toy_graph):
    routes = k_shortest_paths(toy_graph, 1, 4, dist_cost, k=5)
    assert [r.path for r in routes] == [[1, 2, 4], [1, 3, 4]]   # only 2 routes exist
    assert [r.cost for r in routes] == pytest.approx([2.0, 4.0])


def test_tc20_unreachable_goal(toy_graph):
    goal, _, path = SEARCH_METHODS["AS"](problem(toy_graph, 1, 5))
    assert goal is None and path == []
    assert k_shortest_paths(toy_graph, 1, 5, dist_cost, k=5) == []


def test_tc26_heuristic_is_admissible(grid_graph):
    """h(n) (straight-line time at the speed limit) never exceeds the real cheapest cost."""
    def time_cost(edge):                      # distance at 60 km/h, in seconds
        return edge.distance_km / 60 * 3600
    for start in grid_graph.nodes:
        p = TravelTimeProblem(grid_graph, start, [9], time_cost)
        best = min(p.path_cost(path) for path in _all_simple_paths(grid_graph, start, 9))
        assert p.h(start) <= best + 1e-9


def test_tc28_yen_matches_brute_force(grid_graph):
    """Top-5 with AS must equal the 5 cheapest of ALL simple paths (checked exhaustively)."""
    p = problem(grid_graph, 1, 9)
    brute = sorted(p.path_cost(path) for path in _all_simple_paths(grid_graph, 1, 9))[:5]
    routes = k_shortest_paths(grid_graph, 1, 9, dist_cost, k=5, method="AS")
    assert [r.cost for r in routes] == pytest.approx(brute)
    assert len({tuple(r.path) for r in routes}) == 5
    for r in routes:
        assert len(set(r.path)) == len(r.path)
        _valid_path(grid_graph, r.path, 1, 9)


@pytest.mark.parametrize("method", list(SEARCH_METHODS))
def test_top_k_works_with_every_2a_method(method, grid_graph):
    routes = k_shortest_paths(grid_graph, 1, 9, dist_cost, k=5, method=method)
    assert 1 <= len(routes) <= 5
    assert [r.cost for r in routes] == sorted(r.cost for r in routes)
