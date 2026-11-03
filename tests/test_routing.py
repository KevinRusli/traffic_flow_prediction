"""Route finder + end-to-end tests - OWNER: Kevin.  (TC21-TC25, TC27)"""
from datetime import datetime

import pytest

from src.config import resolve_path
from src.prediction.predictor import ConstantPredictor
from src.routing.route_finder import Route, RouteFinder, Segment

WHEN = datetime(2006, 10, 15, 8, 0)


def test_tc21_routes_sorted_with_constant_predictor(toy_graph):
    routes = RouteFinder(toy_graph, ConstantPredictor(100)).find_routes(1, 4, WHEN, k=5)
    assert routes[0].path == [1, 2, 4]
    times = [r.travel_time_s for r in routes]
    assert times == sorted(times)
    # 2 edges x (1 km @ 60 km/h = 60 s + 30 s delay) = 180 s
    assert routes[0].travel_time_s == pytest.approx(180.0, abs=1.0)
    assert routes[0].roads == ["A Rd"]


def test_tc22_invalid_site_raises(toy_graph):
    with pytest.raises(ValueError):
        RouteFinder(toy_graph, ConstantPredictor()).find_routes(1, 9999, WHEN)


def test_tc23_origin_equals_destination(toy_graph):
    routes = RouteFinder(toy_graph, ConstantPredictor()).find_routes(2, 2, WHEN)
    assert routes[0].path == [2] and routes[0].travel_time_s == 0


def test_tc24_end_to_end_spec_example_constant():
    """Real graph, origin 2000 -> destination 3002 (example from the spec)."""
    from src.data.loader import load_long
    from src.graph.build_graph import build_graph

    graph = build_graph(load_long())
    routes = RouteFinder(graph, ConstantPredictor()).find_routes(2000, 3002, WHEN, k=5)
    assert 1 <= len(routes) <= 5


@pytest.mark.parametrize("model", ["lstm", "gru", "third"])
def test_tc25_end_to_end_with_trained_model(model):
    from src.config import load_config

    cfg = load_config()
    if not any(resolve_path(cfg["paths"]["models_dir"]).glob(f"{model}*")):
        pytest.skip(f"no trained '{model}' model in models/saved/ yet")
    from src.data.loader import load_long
    from src.graph.build_graph import build_graph
    from src.prediction.predictor import ModelPredictor

    graph = build_graph(load_long())
    peak = RouteFinder(graph, ModelPredictor(model, cfg)).find_routes(2000, 3002, WHEN)
    night = RouteFinder(graph, ModelPredictor(model, cfg)).find_routes(2000, 3002, WHEN.replace(hour=3))
    assert peak and night
    assert peak[0].travel_time_s >= night[0].travel_time_s, "peak hour should not be faster than 3am"


def test_tc27_route_lists_roads_for_the_gui():
    """Consecutive segments on the same road are merged into one name."""
    route = Route([1, 2, 3, 4, 5], [
        Segment(1, 2, "Warrigal Rd", 1.0, 90), Segment(2, 3, "Warrigal Rd", 0.5, 60),
        Segment(3, 4, "Riversdale Rd", 1.2, 100), Segment(4, 5, "Burke Rd", 0.8, 70),
    ])
    assert route.roads == ["Warrigal Rd", "Riversdale Rd", "Burke Rd"]
    assert route.travel_time_min == pytest.approx(320 / 60)
    assert route.distance_km == pytest.approx(3.5)


class _LastValueModel:
    """Stand-in for a trained model: predicts the last value of each window."""

    def predict(self, X):
        return X[:, -1]


def test_tc30_model_predictor_returns_vehicles_per_hour(toy_long_df):
    from sklearn.preprocessing import MinMaxScaler

    from src.prediction.predictor import ModelPredictor

    scaler = MinMaxScaler().fit(toy_long_df[["flow"]].to_numpy())
    predictor = ModelPredictor("lstm", model=_LastValueModel(), scaler=scaler, df_long=toy_long_df)
    # site 100 at 08:00 -> last input is the 07:45 count = 7 * 10 = 70 per 15 min -> 280 per hour
    assert predictor.predict_flow(100, datetime(2006, 10, 2, 8, 0)) == pytest.approx(280.0)
    # 08:07 falls in the same 15-minute slot
    assert predictor.predict_flow(100, datetime(2006, 10, 2, 8, 7)) == pytest.approx(280.0)
    with pytest.raises(ValueError):
        predictor.predict_flow(999, datetime(2006, 10, 2, 8, 0))
