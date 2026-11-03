"""Graph tests - OWNER: Kevin.  (TC07-TC10)"""
import pytest

from src.data.loader import fix_coordinates, load_raw, to_long
from src.graph.build_graph import build_graph, parse_location


@pytest.fixture(scope="module")
def real_graph(scats_file):
    try:
        return build_graph(fix_coordinates(to_long(load_raw(scats_file))))
    except NotImplementedError as exc:
        pytest.xfail(f"not implemented yet: {exc}")


def _reachable(graph, a, b):
    seen, stack = {a}, [a]
    while stack:
        n = stack.pop()
        if n == b:
            return True
        for e in graph.neighbours(n):
            if e.dst not in seen:
                seen.add(e.dst)
                stack.append(e.dst)
    return False


def test_tc07_parse_location():
    assert parse_location("WARRIGAL_RD N of TOORAK_RD") == ("WARRIGAL_RD", "N", "TOORAK_RD")
    assert parse_location("HIGH_ST NE OF HARP_ST") == ("HIGH_ST", "NE", "HARP_ST")


def test_tc08_graph_nodes_and_degree(real_graph):
    assert len(real_graph) == 40
    lonely = [s for s in real_graph.nodes if not real_graph.neighbours(s)]
    assert not lonely, f"sites without any edge: {lonely}"


def test_tc09_spec_example_reachable(real_graph):
    assert _reachable(real_graph, 2000, 3002)


def test_tc10_graph_structure(toy_graph):
    """Already passes: checks the shared RoadGraph class."""
    assert len(toy_graph) == 5
    assert {e.dst for e in toy_graph.neighbours(1)} == {2, 3}
    assert toy_graph.neighbours(5) == []
    assert toy_graph.haversine_km(1, 2) > 0
