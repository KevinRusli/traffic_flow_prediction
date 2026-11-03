"""TBRGS graphical user interface (Streamlit).

OWNER: Ali Murtaza (GUI)

Run:
    streamlit run app/gui.py        (or: python main.py gui)

What the tutor asked for: like a SIMPLE Google Maps. For each route show the
total minutes and which roads it takes. Nothing more fancy than that.

Required by the spec: user input, parameter settings, visualisation, and
defaults from a configuration file (config.yaml).

Layout (see docs/GUI_SPEC.md for the mockup):
  Sidebar : origin, destination, date + time, model, search method, k
  Main    : map with the routes drawn (fastest = thick, alternatives = thin)
            one card per route: "Route 1 - Fastest - 14 min"
                                "Warrigal Rd -> Riversdale Rd -> Burke Rd"
                                "6.2 km - 5 intersections"

Everything you need is on the Route objects returned by RouteFinder.find_routes():
    route.travel_time_min, route.distance_km, route.roads, route.path, route.segments
    graph.nodes[site].lat / .lon   for the map coordinates

Develop it TODAY with ConstantPredictor and a fake graph; nothing here has to
wait for the ML models.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.search.methods import SEARCH_METHODS  # noqa: E402

cfg = load_config()

st.set_page_config(page_title="TBRGS - Boroondara", layout="wide")
st.title("Traffic-Based Route Guidance System")

with st.sidebar:
    st.header("Trip")
    origin = st.number_input("Origin SCATS site", value=cfg["gui"]["default_origin"], step=1)
    destination = st.number_input("Destination SCATS site", value=cfg["gui"]["default_destination"], step=1)
    default_dt = datetime.strptime(cfg["gui"]["default_datetime"], "%Y-%m-%d %H:%M")
    date = st.date_input("Date", value=default_dt.date())
    time = st.time_input("Time", value=default_dt.time())

    st.header("Settings")
    model_name = st.selectbox("Prediction model", cfg["models"]["available"])
    methods = list(SEARCH_METHODS)
    method = st.selectbox("Search method (from Part A)", methods,
                          index=methods.index(cfg["routing"]["search_method"]))
    k = st.slider("Number of routes", 1, 5, cfg["routing"]["k_paths"])
    find = st.button("Find routes", type="primary")


@st.cache_resource
def load_finder(model: str):
    """Build graph + predictor + RouteFinder once per model.

    TODO(Ali Murtaza):
        from src.data.loader import load_long
        from src.graph.build_graph import build_graph
        from src.prediction.predictor import ConstantPredictor, ModelPredictor
        from src.routing.route_finder import RouteFinder
        graph = build_graph(load_long())
        predictor = ModelPredictor(model, cfg)      # ConstantPredictor() while models aren't trained
        return RouteFinder(graph, predictor, cfg)
    """
    raise NotImplementedError("Ali Murtaza: implement load_finder()")


def draw_map(graph, routes) -> None:
    """TODO(Ali Murtaza): st.pydeck_chart with an OpenStreetMap-style base map.

    - ScatterplotLayer: every SCATS site (graph.nodes[s].lat/.lon), origin and
      destination in a different colour
    - PathLayer: one line per route; route 1 thick + strong colour, others thin + grey
    """
    raise NotImplementedError("Ali Murtaza: implement draw_map()")


def route_card(rank: int, route) -> None:
    """TODO(Ali Murtaza): one bordered container per route.

    Title "Route {rank}" (+ "Fastest" badge for rank 1), big "{minutes} min" on the right,
    below: " -> ".join(route.roads), distance in km, number of intersections
    (len(route.path) - 1). Optional: st.expander with the per-segment table
    (route.segments: road, from site, to site, minutes).
    """
    raise NotImplementedError("Ali Murtaza: implement route_card()")


if find:
    try:
        finder = load_finder(model_name)
        routes = finder.find_routes(int(origin), int(destination),
                                    datetime.combine(date, time), k=k, method=method)
        if not routes:
            st.warning(f"No route found from {origin} to {destination}.")
        else:
            draw_map(finder.graph, routes)
            for rank, route in enumerate(routes, 1):
                route_card(rank, route)
    except ValueError as exc:            # invalid SCATS site etc.
        st.error(str(exc))
    except NotImplementedError as exc:   # template parts not done yet
        st.info(f"Not implemented yet: {exc}")
