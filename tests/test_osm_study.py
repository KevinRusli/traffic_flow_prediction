"""OpenStreetMap research tests (TC31-TC37) - OWNER: Sakib.

Everything runs offline: the OSRM server is replaced by a fake answer, and the route
tests use the toy graph from conftest.py, so the expected numbers can be worked out by hand.
"""
import json

import pandas as pd
import pytest

from src.config import load_config
from src.prediction.predictor import ConstantPredictor
from src.research.osm_study import (band_range, fetch_osm_roads, follows_road, link_congestion, link_key,
                                    offset_study, osm_graph, osm_link_table, parse_osrm, route_effect,
                                    speed_band)

CFG = load_config()


def osrm_answer(km=1.5, names=("Warrigal Road", "Warrigal Road", "")):
    """A minimal OSRM /route answer with one leg."""
    return {"code": "Ok", "routes": [{
        "distance": km * 1000, "duration": 120.0,
        "geometry": {"coordinates": [[145.0, -37.8], [145.005, -37.801], [145.01, -37.8]]},
        "legs": [{"steps": [{"name": n} for n in names]}]}]}


def free_flow(edge):
    """60 km/h on every edge + 30 s delay: 1 km = 90 s."""
    return edge.distance_km * 60 + 30


class SitePredictor:
    """Constant flow everywhere except at the sites given in `flows`."""

    def __init__(self, flows, default=200.0):
        self.flows, self.default = flows, default

    def predict_flow(self, scats, when):
        return self.flows.get(scats, self.default)


def test_tc31_speed_bands():
    assert speed_band(60.0)[0] == "Free flow"
    assert speed_band(55.0)[0] == "Free flow"           # a band includes its lower limit
    assert speed_band(52.0)[0] == "Slowed"
    assert speed_band(44.6)[0] == "Heavy"
    assert band_range(0.0) == "< 50 km/h" and band_range(50.0) == "50-55 km/h"


def test_tc32_offset_study_recovers_a_known_offset():
    raw = pd.DataFrame({"scats": [1, 1, 2, 2], "lat": [-37.8, -37.801, -37.9, 0.0],
                        "lon": [145.0, 145.001, 145.1, 0.0],
                        "location": ["A_RD N of B_RD", "B_RD E of A_RD", "C_RD S of D_RD", "D_RD W of C_RD"]})
    counts = pd.DataFrame({"TFM_DESC": ["A RD N OF B RD", "B RD E OF A RD", "C RD S OF D RD", "OTHER RD N OF X RD"],
                           "Y": [-37.8 + 0.0015, -37.801 + 0.0015, -37.9 + 0.0015, -36.0],
                           "X": [145.0 + 0.0013, 145.001 + 0.0013, 145.1 + 0.0013, 144.0]})
    matched, summary = offset_study(raw, counts, {"lat": 0.0015, "lon": 0.0013})
    assert summary["matched"] == 3 and summary["approaches"] == 3      # the (0, 0) approach is left out
    assert summary["fitted_lat"] == pytest.approx(0.0015) and summary["fitted_lon"] == pytest.approx(0.0013)
    assert summary["median_before_m"] > 150                            # about 200 m off before
    assert matched.error_after_m.max() == pytest.approx(0.0, abs=0.01)


def test_tc33_link_congestion_uses_the_slower_direction(toy_graph):
    table = link_congestion(toy_graph, ConstantPredictor(200.0), None, CFG)
    assert len(table) == 4                                             # one row per two-way link
    assert (table.speed == 60.0).all() and (table.band == "Free flow").all()

    jammed = link_congestion(toy_graph, SitePredictor({4: 1300.0}), None, CFG)
    link = jammed[(jammed.a == 2) & (jammed.b == 4)].iloc[0]
    assert link.speed_ab < 45 and link.speed_ba == 60.0                # only 2 -> 4 drives into site 4
    assert link.band == "Heavy" and link.path == [[145.01, -37.8], [145.02, -37.8]]


def test_tc34_parse_osrm_answer():
    road = parse_osrm(osrm_answer(km=1.5))
    assert road["km"] == pytest.approx(1.5) and road["minutes"] == pytest.approx(2.0)
    assert road["names"] == ["Warrigal Road"] and len(road["path"]) == 3
    with pytest.raises(ValueError):
        parse_osrm({"code": "NoRoute", "routes": []})


def test_tc35_fetch_saves_every_link_and_resumes(toy_graph, tmp_path):
    out = tmp_path / "osm_roads.json"
    calls = []
    roads = fetch_osm_roads(toy_graph, out, get_json=lambda url: calls.append(url) or osrm_answer(), pause_s=0)
    assert len(calls) == 4 and set(roads) == {"1-2", "1-3", "2-4", "3-4"}
    assert json.loads(out.read_text()) == roads

    def no_internet(url):
        raise AssertionError("every link is cached, so nothing should be downloaded")
    assert fetch_osm_roads(toy_graph, out, get_json=no_internet, pause_s=0) == roads


def test_tc36_road_check_and_detour_factor(toy_graph):
    assert follows_road("Warrigal Rd", ["Warrigal Road"])
    assert follows_road("Eastern Fwy", ["Eastern Freeway"])
    assert not follows_road("Warrigal Rd", ["Toorak Road", ""])
    roads = {link_key(1, 2): parse_osrm(osrm_answer(km=1.76, names=("A Road",)))}
    table = osm_link_table(toy_graph, roads)
    assert len(table) == 1 and bool(table.iloc[0].follows_road)
    assert table.iloc[0].detour == pytest.approx(1.76 / toy_graph.haversine_km(1, 2))


def test_tc37_real_road_lengths_change_the_route(toy_graph):
    # the real 1-2 road is 5 km, so 1-2-4 (6 km) loses to 1-3-4 (4 km)
    roads = {link_key(1, 2): parse_osrm(osrm_answer(km=5.0)), link_key(1, 3): parse_osrm(osrm_answer(km=0.1))}
    real = osm_graph(toy_graph, roads)
    assert real.neighbours(1)[0].distance_km == pytest.approx(5.0)
    assert min(e.distance_km for e in real.neighbours(3) if e.dst == 1) == pytest.approx(1.0)  # never below straight
    effect = route_effect(toy_graph, real, (free_flow, free_flow), [(1, 4)])
    row = effect.iloc[0]
    assert bool(row.changed)
    assert row.straight_min == pytest.approx(180 / 60) and row.real_min == pytest.approx(300 / 60)
    assert row.time_lost_s == pytest.approx(420 - 300)                # 1-2-4 driven on the real roads
