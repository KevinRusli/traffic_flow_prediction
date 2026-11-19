"""TBRGS GUI (Streamlit).

Run: streamlit run app/gui.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.loader import load_long  # noqa: E402
from src.graph.build_graph import build_graph  # noqa: E402
from src.prediction.predictor import ConstantPredictor, ModelPredictor  # noqa: E402
from src.routing.route_finder import RouteFinder  # noqa: E402
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
    graph = build_graph(load_long())
    try:
        predictor = ModelPredictor(model, cfg)
    except FileNotFoundError:
        st.warning(f"No trained '{model}' model found yet - using ConstantPredictor for now.")
        predictor = ConstantPredictor()
    return RouteFinder(graph, predictor, cfg)


def draw_map(graph, routes) -> None:
    site_points = [
        {"lat": node.lat, "lon": node.lon, "label": str(sid), "color": [100, 100, 100]}
        for sid, node in graph.nodes.items()
    ]
    # green = origin, red = destination, so they're easy to spot on the map
    for sid, color in ((routes[0].path[0], [0, 160, 0]), (routes[0].path[-1], [200, 0, 0])):
        for pt in site_points:
            if pt["label"] == str(sid):
                pt["color"] = color

    path_layers = []
    for rank, route in enumerate(routes, 1):
        coords = [[graph.nodes[s].lon, graph.nodes[s].lat] for s in route.path]
        is_fastest = rank == 1
        path_layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": coords}],
            get_path="path",
            get_width=6 if is_fastest else 2,
            get_color=[0, 120, 255] if is_fastest else [160, 160, 160],
            width_min_pixels=6 if is_fastest else 2,
        ))

    site_layer = pdk.Layer(
        "ScatterplotLayer",
        data=site_points,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=40,
        pickable=True,
    )

    origin_node = graph.nodes[routes[0].path[0]]
    view_state = pdk.ViewState(latitude=origin_node.lat, longitude=origin_node.lon, zoom=12)

    st.pydeck_chart(pdk.Deck(
        layers=[site_layer, *path_layers],
        initial_view_state=view_state,
        map_style="road",
        tooltip={"text": "Site {label}"},
    ))


def route_card(rank: int, route) -> None:
    with st.container(border=True):
        title_col, time_col = st.columns([4, 1])
        with title_col:
            label = f"Route {rank}"
            if rank == 1:
                label += "  🟢 Fastest"
            st.markdown(f"**{label}**")
        with time_col:
            st.markdown(f"**{route.travel_time_min:.0f} min**")

        st.write(" → ".join(route.roads) if route.roads else " → ".join(str(s) for s in route.path))
        st.caption(f"{route.distance_km:.1f} km · {len(route.path) - 1} intersections")

        if route.segments:
            with st.expander("Segment detail"):
                st.table([
                    {
                        "From": seg.src,
                        "To": seg.dst,
                        "Road": seg.road,
                        "Distance (km)": round(seg.distance_km, 2),
                        "Time (min)": round(seg.time_s / 60, 1),
                    }
                    for seg in route.segments
                ])


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
    except ValueError as exc:
        st.error(str(exc))
