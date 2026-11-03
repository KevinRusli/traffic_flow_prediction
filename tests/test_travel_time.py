"""Flow -> speed -> travel time tests - OWNER: Hassan.  (TC14-TC16)"""
import pytest

from src.routing.travel_time import edge_travel_time, flow_to_speed


@pytest.mark.parametrize("flow, expected", [(0, 60.0), (100, 60.0), (351, 60.0), (1500, 32.0)])
def test_tc14_flow_to_speed_reference_points(flow, expected):
    assert flow_to_speed(flow) == pytest.approx(expected, abs=0.1)


def test_tc15_edge_time_free_flow():
    # 1 km at 60 km/h = 60 s, plus 30 s intersection delay
    assert edge_travel_time(1.0, 100) == pytest.approx(90.0, abs=0.5)


def test_tc16_speed_decreases_with_flow():
    speeds = [flow_to_speed(f) for f in range(400, 1501, 100)]
    assert all(a > b for a, b in zip(speeds, speeds[1:]))
    assert all(32 - 0.1 <= s <= 60 for s in speeds)
