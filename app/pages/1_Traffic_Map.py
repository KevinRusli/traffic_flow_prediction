"""Traffic Map page (Research): predicted congestion on every road, like Google Maps Traffic.

OWNER: Sakib

Opens as a second page of the GUI: streamlit run app/gui.py -> "Traffic Map" in the sidebar.
Each link is coloured by the speed its predicted flow gives (src/research/osm_study.py).
With the OpenStreetMap shapes fetched (python -m src.research.osm_study fetch) the links follow
the real roads; the raw SCATS positions can be shown to see the coordinate correction.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.loader import load_long, load_raw  # noqa: E402
from src.graph.build_graph import build_graph  # noqa: E402
from src.prediction.predictor import ModelPredictor  # noqa: E402
from src.research.osm_study import (MODEL_LABELS, OSM_ROADS_FILE, OUT_DIR, SPEED_BANDS, STUDY_DAY,  # noqa: E402
                                    band_range, link_congestion, raw_site_positions)

cfg = load_config()
st.set_page_config(page_title="TBRGS - Traffic Map", layout="wide")
st.title("Traffic Map")
st.caption("Predicted speed on every road link, coloured like Google Maps Traffic.")

roads_file = resolve_path(OUT_DIR) / OSM_ROADS_FILE
# every 15-minute slot of a day, the same steps as the dataset
slots = [time(h, m) for h in range(24) for m in range(0, 60, cfg["data"]["interval_minutes"])]

with st.sidebar:
    st.header("When")
    date = st.date_input("Date", value=STUDY_DAY.date())
    when = st.select_slider("Time", options=slots, value=time(8, 0), format_func=lambda t: t.strftime("%H:%M"))
    st.header("Settings")
    model_name = st.selectbox("Prediction model", cfg["models"]["available"],
                              index=cfg["models"]["available"].index("third"),
                              format_func=lambda m: MODEL_LABELS.get(m, m))
    real_shapes = st.checkbox("Real road shapes (OpenStreetMap)", value=roads_file.exists(),
                              disabled=not roads_file.exists(),
                              help="Run `python -m src.research.osm_study fetch` once to download them.")
    show_raw = st.checkbox("Show raw SCATS positions (before the offset correction)")


@st.cache_resource
def load_network():
    """Road graph and the long data, loaded once per session."""
    df_long = load_long()
    return build_graph(df_long), df_long


@st.cache_resource
def load_predictor(name: str):
    graph, df_long = load_network()
    return ModelPredictor(name, cfg, df_long=df_long)


@st.cache_data
def load_raw_positions() -> pd.DataFrame:
    return raw_site_positions(load_raw())


graph, _ = load_network()
try:
    predictor = load_predictor(model_name)
except Exception as exc:                          # e.g. the model has not been trained on this PC
    st.error(f"Could not load the {MODEL_LABELS.get(model_name, model_name)} model: {exc}")
    st.stop()

roads = json.loads(roads_file.read_text()) if real_shapes else None
moment = datetime.combine(date, when)
table = link_congestion(graph, predictor, moment, cfg, roads)

# ---- summary numbers above the map ----
cols = st.columns(len(SPEED_BANDS) + 1)
cols[0].metric("Mean link speed", f"{table.speed.mean():.1f} km/h")
for col, (lower, band, _) in zip(cols[1:], SPEED_BANDS):
    col.metric(f"{band} ({band_range(lower)})", f"{(table.band == band).sum()} links")

# ---- map layers ----
table["label"] = table.apply(lambda r: f"{r.road}: {r.a}->{r.b} {r.speed_ab:.0f} km/h, "
                                       f"{r.b}->{r.a} {r.speed_ba:.0f} km/h", axis=1)
links_layer = pdk.Layer("PathLayer", data=table[["path", "colour", "label"]], get_path="path",
                        get_color="colour", width_min_pixels=5, pickable=True)
sites = pd.DataFrame([{"lat": n.lat, "lon": n.lon, "label": f"Site {s}: {n.name}"} for s, n in graph.nodes.items()])
sites_layer = pdk.Layer("ScatterplotLayer", data=sites, get_position=["lon", "lat"], get_fill_color=[40, 40, 40],
                        get_radius=35, pickable=True)
layers = [links_layer, sites_layer]

if show_raw:
    # grey dot = raw SCATS position, thin line = the correction to the drawn position
    fixed = sites.assign(scats=list(graph.nodes)).set_index("scats")
    raw = load_raw_positions().merge(fixed, left_index=True, right_index=True, suffixes=("", "_fixed"))
    raw["label"] = "Raw position of " + raw["label"]
    raw["line"] = raw.apply(lambda r: [[r.lon, r.lat], [r.lon_fixed, r.lat_fixed]], axis=1)
    layers.append(pdk.Layer("PathLayer", data=raw[["line"]], get_path="line", get_color=[90, 90, 90],
                            width_min_pixels=1))
    layers.append(pdk.Layer("ScatterplotLayer", data=raw[["lat", "lon", "label"]], get_position=["lon", "lat"],
                            get_fill_color=[150, 150, 150], get_radius=30, pickable=True))

view = pdk.ViewState(latitude=sites.lat.mean(), longitude=sites.lon.mean(), zoom=12.3)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="road", tooltip={"text": "{label}"}))
st.caption(" · ".join(f"{band}: {band_range(lo)}" for lo, band, _ in SPEED_BANDS)
           + ". A link takes the colour of its slower direction.")

# ---- slowest links ----
st.subheader("Slowest links")
slowest = table.nsmallest(10, "speed")[["road", "a", "b", "speed_ab", "speed_ba", "flow_ab", "flow_ba"]]
st.dataframe(slowest.round(1).rename(columns={
    "road": "Road", "a": "Site A", "b": "Site B", "speed_ab": "A->B km/h", "speed_ba": "B->A km/h",
    "flow_ab": "A->B veh/h", "flow_ba": "B->A veh/h"}), hide_index=True)
