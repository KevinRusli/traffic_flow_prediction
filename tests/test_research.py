"""Research experiment tests - OWNER: Kevin.

The study code is checked on the toy graphs, with costs in seconds at 60 km/h
plus the 30 s intersection delay, so the expected numbers can be worked out by hand.
"""
import pytest

from src.research.search_study import (compare_methods, compare_route_choice, fastest_time, site_pairs,
                                       summarise_methods)
from src.search.methods import SEARCH_METHODS


def free_flow(edge):
    """60 km/h on every edge + 30 s delay: 1 km = 90 s."""
    return edge.distance_km * 60 + 30


def jam_on_2_to_4(edge):
    """Same as free flow, but the edge 2 -> 4 takes 1000 s."""
    return 1000.0 if (edge.src, edge.dst) == (2, 4) else free_flow(edge)


def test_site_pairs_are_every_ordered_pair_of_different_sites(grid_graph):
    pairs = site_pairs(grid_graph)
    assert len(pairs) == 9 * 8 and len(set(pairs)) == len(pairs)
    assert all(a != b for a, b in pairs)


def test_dijkstra_reference_time(toy_graph):
    assert fastest_time(toy_graph, 1, 4, free_flow) == pytest.approx(180.0)     # 1-2-4
    assert fastest_time(toy_graph, 1, 4, jam_on_2_to_4) == pytest.approx(300.0)  # 1-3-4
    assert fastest_time(toy_graph, 1, 5, free_flow) is None                     # 5 is isolated


def test_methods_study_astar_matches_dijkstra(grid_graph):
    detail = compare_methods(grid_graph, free_flow, site_pairs(grid_graph))
    assert (detail.extra_pct >= -1e-6).all()            # no method beats the fastest time
    table = summarise_methods(detail)
    assert list(table.index) == list(SEARCH_METHODS)
    assert table.loc["AS", "optimal_pct"] == pytest.approx(100.0)
    assert table.loc["AS", "max_extra_pct"] == pytest.approx(0.0)


def test_route_choice_counts_changed_routes_and_time_lost(toy_graph):
    # Without the jam the fastest 1 -> 4 is 1-2-4 (180 s). With it, 1-3-4 (90 + 210 = 300 s).
    # Driving 1-2-4 in the jam costs 90 + 1000 = 1090 s, so 790 s are lost.
    table = compare_route_choice(toy_graph, {"jam": jam_on_2_to_4, "free": free_flow}, "jam", [(1, 4)])
    assert table.loc["jam", "changed_pct"] == 0.0
    assert table.loc["free", "changed_pct"] == 100.0
    assert table.loc["free", "mean_lost_s"] == pytest.approx(790.0)


def test_unreachable_trips_are_left_out(toy_graph):
    # site 5 has no edges, so only the trip 1 -> 4 is routed
    detail = compare_methods(toy_graph, free_flow, [(1, 5), (1, 4)])
    assert set(zip(detail.origin, detail.goal)) == {(1, 4)}
