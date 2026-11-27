"""Research: show the predictions and routes on OpenStreetMap (Google Maps Traffic idea).

OWNER: Sakib

This is research initiative 3 of the A2B brief: visualise the predictions and route
recommendations with open-source map data, and fix the SCATS latitude/longitude problem
the brief warns about. It has four parts:

1. Coordinate correction. The SCATS coordinates are offset from the real intersections.
   Every SCATS approach is matched by name to the VicRoads count points
   (data/raw/Traffic_Count_Locations_with_LONG_LAT.csv, real coordinates), which gives
   the offset from all matched approaches instead of a few sites, and the error in metres
   before and after the correction.
2. Traffic layer. Every road link is coloured by the speed its predicted flow gives
   (green / amber / red), like Google Maps Traffic, for any time and model.
3. Real road shapes. For every link, OSRM (a router that uses OpenStreetMap data) returns
   the real driving path, its length and the road names it follows. This checks the road
   graph that was inferred from the location names and measures how much longer real
   roads are than the straight lines the graph uses.
4. Route effect. All 1,560 trips are routed again with the real road lengths to see how
   often the recommended route changes.

Run once on a PC with internet:   python -m src.research.osm_study fetch
Then (no internet needed):         python -m src.research.osm_study run
Map page:                          streamlit run app/gui.py  -> "Traffic Map" in the sidebar
Outputs: results/research_osm/
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.graph.graph import RoadGraph
from src.routing.route_finder import path_time, site_pairs
from src.routing.travel_time import flow_to_speed
from src.search.top_k import k_shortest_paths

matplotlib.use("Agg")                                # write PNG files, no window
from matplotlib import pyplot as plt  # noqa: E402

OUT_DIR = "results/research_osm"
OSM_ROADS_FILE = "osm_roads.json"                    # cached OSRM answers, committed with the results
COUNTS_CSV = "data/raw/Traffic_Count_Locations_with_LONG_LAT.csv"
OSRM_URL = ("https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
            "?overview=full&geometries=geojson&steps=true")
OSRM_PAUSE_S = 1.0                                   # the public OSRM server allows about 1 request per second
EARTH_RADIUS_M = 6_371_000
# Some VicRoads count points sit mid-block, a few hundred metres from the intersection they are
# named after, so the summary also counts how many approaches end up close to their count point.
NEAR_M = 25.0

# Colour bands by predicted speed (km/h). With the conversion PDF, 55 km/h is about 725 veh/h
# and 50 km/h about 1,025 veh/h, so green = light traffic, red = heavy traffic.
SPEED_BANDS = [(55.0, "Free flow", [0, 160, 80]),
               (50.0, "Slowed", [255, 170, 0]),
               (0.0, "Heavy", [220, 30, 30])]

STUDY_DAY = datetime(2006, 10, 30)                   # Monday of the test week, same as the other study
STUDY_HOURS = [3, 8, 17]                             # quiet night, morning peak, evening peak
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "third": "Random Forest"}

# Words OpenStreetMap writes in full but the SCATS names abbreviate.
_ROAD_WORDS = {"rd": "road", "st": "street", "hwy": "highway", "fwy": "freeway", "gv": "grove",
               "pde": "parade", "av": "avenue", "ave": "avenue", "dr": "drive", "cr": "crescent"}


# ---- helpers ----------------------------------------------------------------------------
def distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; works on numbers or numpy/pandas arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


def link_key(a: int, b: int) -> str:
    """One key per two-way link, smaller site first: (3002, 2000) -> "2000-3002"."""
    return f"{min(a, b)}-{max(a, b)}"


def undirected_links(graph: RoadGraph) -> list[tuple[int, int, str]]:
    """Every two-way road link once, as (smaller site, larger site, road name)."""
    return sorted({(min(e.src, e.dst), max(e.src, e.dst), e.road) for e in graph.edges()})


def _normalise_name(text: str) -> str:
    """'WARRIGAL_RD N of TOORAK_RD #2' -> 'WARRIGAL RD N OF TOORAK RD', so both datasets compare equal."""
    text = re.sub(r"\s+#\d+$", "", str(text).upper().replace("_", " "))
    return re.sub(r"\s+", " ", text).strip()


# ---- 1. coordinate correction ----------------------------------------------------------------
def offset_study(raw: pd.DataFrame, counts: pd.DataFrame, offset: dict) -> tuple[pd.DataFrame, dict]:
    """Compare the raw SCATS coordinates with the real VicRoads coordinates.

    raw:    load_raw() output (columns scats, location, lat, lon) before any correction.
    counts: the VicRoads count points (columns TFM_DESC, X = longitude, Y = latitude).
    offset: the correction used by the loader, {"lat": .., "lon": ..}.
    Returns one row per matched approach and a summary of the errors in metres.
    """
    known = raw[(raw["lat"] != 0) & (raw["lon"] != 0)]          # site 4266 has one approach at 0, 0
    approaches = known.drop_duplicates("location")[["scats", "location", "lat", "lon"]].copy()
    approaches["key"] = approaches["location"].map(_normalise_name)

    # a description can appear several times (one row per count year), so take its median position
    counts = counts.assign(key=counts["TFM_DESC"].map(_normalise_name))
    real = counts.groupby("key")[["Y", "X"]].median().rename(columns={"Y": "real_lat", "X": "real_lon"})
    matched = approaches.join(real, on="key", how="inner")

    matched["error_before_m"] = distance_m(matched.lat, matched.lon, matched.real_lat, matched.real_lon)
    matched["error_after_m"] = distance_m(matched.lat + offset["lat"], matched.lon + offset["lon"],
                                          matched.real_lat, matched.real_lon)
    summary = {
        "approaches": int(len(approaches)), "matched": int(len(matched)),
        "sites_matched": int(matched["scats"].nunique()), "sites": int(approaches["scats"].nunique()),
        # the median is used so that a few wrongly matched count points cannot move the estimate
        "fitted_lat": float((matched.real_lat - matched.lat).median()),
        "fitted_lon": float((matched.real_lon - matched.lon).median()),
        "median_before_m": float(matched.error_before_m.median()),
        "median_after_m": float(matched.error_after_m.median()),
        "p90_before_m": float(matched.error_before_m.quantile(0.9)),
        "p90_after_m": float(matched.error_after_m.quantile(0.9)),
        "near_before": int((matched.error_before_m <= NEAR_M).sum()),
        "near_after": int((matched.error_after_m <= NEAR_M).sum()),
    }
    return matched.drop(columns="key"), summary


def raw_site_positions(raw: pd.DataFrame) -> pd.DataFrame:
    """Uncorrected position of each site (median of its approaches), to show the correction on the map."""
    known = raw[(raw["lat"] != 0) & (raw["lon"] != 0)]
    return known.groupby("scats")[["lat", "lon"]].median()


# ---- 2. traffic layer ----------------------------------------------------------------------
def speed_band(speed_kmh: float) -> tuple[str, list[int]]:
    """Name and RGB colour of the band a speed falls in (SPEED_BANDS)."""
    for lower, name, colour in SPEED_BANDS:
        if speed_kmh >= lower:
            return name, colour
    return SPEED_BANDS[-1][1], SPEED_BANDS[-1][2]


def band_range(lower: float) -> str:
    """Readable speed range of the band that starts at `lower`, e.g. '50-55 km/h'."""
    lowers = [lo for lo, _, _ in SPEED_BANDS]
    upper = lowers[lowers.index(lower) - 1] if lowers.index(lower) > 0 else None
    if upper is None:
        return f">= {lower:.0f} km/h"
    return f"< {upper:.0f} km/h" if lower == 0 else f"{lower:.0f}-{upper:.0f} km/h"


def link_congestion(graph: RoadGraph, predictor, when: datetime, cfg: dict,
                    roads: dict | None = None) -> pd.DataFrame:
    """One row per two-way link with the predicted speed in each direction and its colour.

    As in routing, the flow of a direction A -> B is the predicted flow at site B.
    The link is coloured by its slower direction, so a jam is never hidden.
    roads (OSRM answers) gives the real road shape; without it the link is a straight line.
    """
    rows = []
    for a, b, road in undirected_links(graph):
        speeds = {}
        for src, dst in ((a, b), (b, a)):
            flow = predictor.predict_flow(dst, when)
            speeds[f"{src}->{dst}"] = (flow, flow_to_speed(flow, cfg))
        (flow_ab, speed_ab), (flow_ba, speed_ba) = speeds.values()
        band, colour = speed_band(min(speed_ab, speed_ba))
        rows.append(dict(a=a, b=b, road=road, km=graph.haversine_km(a, b),
                         flow_ab=flow_ab, speed_ab=speed_ab, flow_ba=flow_ba, speed_ba=speed_ba,
                         speed=min(speed_ab, speed_ba), band=band, colour=colour,
                         path=link_path(graph, a, b, roads)))
    return pd.DataFrame(rows)


def link_path(graph: RoadGraph, a: int, b: int, roads: dict | None = None) -> list[list[float]]:
    """[[lon, lat], ...] of a link: the OpenStreetMap shape if fetched, otherwise a straight line."""
    if roads and link_key(a, b) in roads:
        return roads[link_key(a, b)]["path"]
    return [[graph.nodes[s].lon, graph.nodes[s].lat] for s in (a, b)]


def congestion_summary(graph: RoadGraph, predictors: dict, hours: list[int], cfg: dict) -> pd.DataFrame:
    """Per model and hour: mean and lowest link speed, and how many links fall in each band."""
    rows = []
    for name, predictor in predictors.items():
        for hour in hours:
            table = link_congestion(graph, predictor, STUDY_DAY.replace(hour=hour), cfg)
            row = dict(model=MODEL_LABELS.get(name, name), hour=f"{hour:02d}:00",
                       mean_speed=table.speed.mean(), min_speed=table.speed.min())
            for _, band, _ in SPEED_BANDS:
                row[band] = int((table.band == band).sum())
            rows.append(row)
    return pd.DataFrame(rows)


# ---- 3. real road shapes from OpenStreetMap ---------------------------------------------------
def _download_json(url: str) -> dict:
    """GET a URL and parse its JSON answer (OSRM asks clients to name themselves)."""
    request = urllib.request.Request(url, headers={"User-Agent": "COS30019-TBRGS-student-project"})
    with urllib.request.urlopen(request, timeout=30) as answer:
        return json.load(answer)


def parse_osrm(answer: dict) -> dict:
    """Keep what the study needs from an OSRM route answer: length, time, shape and road names."""
    if answer.get("code") != "Ok" or not answer.get("routes"):
        raise ValueError(f"OSRM found no route: {answer.get('code')}")
    route = answer["routes"][0]
    names = [step["name"] for leg in route["legs"] for step in leg.get("steps", []) if step.get("name")]
    return dict(km=route["distance"] / 1000, minutes=route["duration"] / 60,
                path=route["geometry"]["coordinates"], names=sorted(set(names)))


def fetch_osm_roads(graph: RoadGraph, out_file: Path, get_json: Callable[[str], dict] = _download_json,
                    pause_s: float = OSRM_PAUSE_S) -> dict:
    """Ask OSRM for the real driving path of every link and save the answers to out_file.

    Links already in out_file are skipped, so a stopped run can simply be started again.
    get_json is a parameter so the tests can pass a fake server instead of the internet.
    """
    roads = json.loads(out_file.read_text()) if out_file.exists() else {}
    todo = [(a, b) for a, b, _ in undirected_links(graph) if link_key(a, b) not in roads]
    for i, (a, b) in enumerate(todo, 1):
        na, nb = graph.nodes[a], graph.nodes[b]
        url = OSRM_URL.format(lon1=na.lon, lat1=na.lat, lon2=nb.lon, lat2=nb.lat)
        roads[link_key(a, b)] = parse_osrm(get_json(url))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(roads))            # save after every link
        print(f"  {i}/{len(todo)} {link_key(a, b)}: {roads[link_key(a, b)]['km']:.2f} km")
        time.sleep(pause_s)
    return roads


def _road_core(name: str) -> str:
    """'Warrigal Rd' -> 'warrigal road', 'Eastern Fwy' -> 'eastern freeway', for comparing road names."""
    words = re.sub(r"[^a-z ]", " ", name.lower()).split()
    return " ".join(_ROAD_WORDS.get(w, w) for w in words)


def follows_road(our_road: str, osm_names: list[str]) -> bool:
    """True if OpenStreetMap's path uses the road our graph says links the two sites."""
    ours = _road_core(our_road)
    return any(ours in _road_core(n) or _road_core(n) in ours for n in osm_names if _road_core(n))


def osm_link_table(graph: RoadGraph, roads: dict) -> pd.DataFrame:
    """Per link: straight-line km, real road km, how much longer (detour factor) and the road check."""
    rows = []
    for a, b, road in undirected_links(graph):
        if link_key(a, b) not in roads:
            continue
        osm = roads[link_key(a, b)]
        straight = graph.haversine_km(a, b)
        rows.append(dict(a=a, b=b, road=road, straight_km=straight, osm_km=osm["km"],
                         detour=osm["km"] / straight, osm_minutes=osm["minutes"],
                         follows_road=follows_road(road, osm["names"]), osm_names="; ".join(osm["names"])))
    return pd.DataFrame(rows)


# ---- 4. route effect ------------------------------------------------------------------------------
def osm_graph(graph: RoadGraph, roads: dict) -> RoadGraph:
    """Same sites and links, but each link is as long as its real road.

    A link is never made shorter than its straight line: the A* heuristic assumes no road is
    shorter than the straight line, and keeping that true keeps A* optimal.
    """
    real = RoadGraph()
    for node in graph.nodes.values():
        real.add_node(node)
    for e in graph.edges():
        osm = roads.get(link_key(e.src, e.dst))
        km = max(osm["km"], e.distance_km) if osm else e.distance_km
        real.add_edge(e.src, e.dst, km, e.road)
    return real


def route_effect(straight: RoadGraph, real: RoadGraph, cost_fns: tuple, pairs,
                 speed_limit_kmh: float = 60.0) -> pd.DataFrame:
    """Route every trip on both graphs with A* and compare the recommended routes.

    cost_fns = (cost on the straight-line graph, cost on the real-road graph); both use the
    same predicted flows, only the edge lengths differ. time_lost_s is how much longer the
    straight-line recommendation takes when it is driven on the real roads.
    """
    straight_cost, real_cost = cost_fns
    rows = []
    for origin, goal in pairs:
        old = k_shortest_paths(straight, origin, goal, straight_cost, 1, "AS", speed_limit_kmh)
        new = k_shortest_paths(real, origin, goal, real_cost, 1, "AS", speed_limit_kmh)
        if not old or not new:
            continue
        old_on_real = path_time(real, old[0].path, real_cost)
        rows.append(dict(origin=origin, goal=goal, changed=old[0].path != new[0].path,
                         straight_min=old[0].cost / 60, real_min=new[0].cost / 60,
                         time_lost_s=old_on_real - new[0].cost))
    return pd.DataFrame(rows)


# ---- figures --------------------------------------------------------------------------------------
def plot_congestion_maps(graph: RoadGraph, tables: dict, out_file: Path) -> None:
    """One small map per (label -> link_congestion table), links coloured by band."""
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 5), sharey=True)
    for ax, (label, table) in zip(np.atleast_1d(axes), tables.items()):
        for row in table.itertuples():
            lons, lats = zip(*row.path)
            ax.plot(lons, lats, color=np.array(row.colour) / 255, linewidth=2.5)
        ax.scatter([n.lon for n in graph.nodes.values()], [n.lat for n in graph.nodes.values()],
                   s=8, color="black", zorder=3)
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.ticklabel_format(useOffset=False)                 # show 145.02, not 0.02 + 1.45e2
        ax.set_aspect(1 / math.cos(math.radians(-37.82)))   # equal distances in both directions
    np.atleast_1d(axes)[0].set_ylabel("Latitude")
    handles = [plt.Line2D([], [], color=np.array(c) / 255, linewidth=3, label=f"{n} ({band_range(lo)})")
               for lo, n, c in SPEED_BANDS]
    fig.legend(handles=handles, loc="lower center", ncol=len(SPEED_BANDS))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_offset_errors(matched: pd.DataFrame, out_file: Path) -> None:
    """Histogram of the distance to the real intersection, before and after the correction."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, max(matched.error_before_m.quantile(0.95), 50), 30)
    ax.hist(matched.error_before_m, bins=bins, alpha=0.6, label="Raw SCATS coordinates")
    ax.hist(matched.error_after_m, bins=bins, alpha=0.6, label="After the offset correction")
    ax.set_xlabel("Distance to the real intersection (m)")
    ax.set_ylabel("Approaches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ---- command line -----------------------------------------------------------------------------------
def _load_network():
    """Config, long data, raw data and road graph (imported here so the tests stay light)."""
    from src.data.loader import load_long, load_raw
    from src.graph.build_graph import build_graph

    cfg = load_config()
    df_long = load_long()
    return cfg, df_long, load_raw(), build_graph(df_long)


def _load_predictors(cfg: dict, df_long: pd.DataFrame) -> dict:
    """Every trained model that loads; a model that does not load is skipped with a message."""
    from src.prediction.predictor import ModelPredictor

    predictors = {}
    for name in cfg["models"]["available"]:
        try:
            predictors[name] = ModelPredictor(name, cfg, df_long=df_long)
        except Exception as exc:                      # e.g. TensorFlow not installed on this PC
            print(f"  skipped {name}: {exc}")
    return predictors


def run(out_dir: Path | None = None) -> None:
    """Run parts 1, 2 and (if the OSRM answers were fetched) 3 and 4, and write the results."""
    from src.routing.route_finder import RouteFinder

    out = out_dir or resolve_path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    cfg, df_long, raw, graph = _load_network()
    notes = []

    # 1. coordinate correction
    counts = pd.read_csv(resolve_path(COUNTS_CSV), usecols=["X", "Y", "TFM_DESC"])
    matched, off = offset_study(raw, counts, cfg["data"]["coord_offset"])
    matched.to_csv(out / "offset_errors.csv", index=False)
    plot_offset_errors(matched, out / "offset_errors.png")
    notes.append(f"Offset: {off['matched']}/{off['approaches']} approaches matched "
                 f"({off['sites_matched']}/{off['sites']} sites); fitted offset "
                 f"{off['fitted_lat']:+.5f} lat, {off['fitted_lon']:+.5f} lon; median error "
                 f"{off['median_before_m']:.0f} m -> {off['median_after_m']:.0f} m, 90th percentile "
                 f"{off['p90_before_m']:.0f} m -> {off['p90_after_m']:.0f} m; within {NEAR_M:.0f} m: "
                 f"{off['near_before']} -> {off['near_after']} approaches")

    # 2. traffic layer for every model at night, morning peak and evening peak
    predictors = _load_predictors(cfg, df_long)
    roads_file = out / OSM_ROADS_FILE
    roads = json.loads(roads_file.read_text()) if roads_file.exists() else None
    summary = congestion_summary(graph, predictors, STUDY_HOURS, cfg)
    summary.to_csv(out / "congestion_summary.csv", index=False)
    notes.append("Congestion (links per band):\n" + summary.round(1).to_string(index=False))
    reference = "third" if "third" in predictors else next(iter(predictors))
    tables = {f"{MODEL_LABELS[reference]} {h:02d}:00": link_congestion(
        graph, predictors[reference], STUDY_DAY.replace(hour=h), cfg, roads) for h in STUDY_HOURS}
    plot_congestion_maps(graph, tables, out / "congestion_maps.png")

    # 3. and 4. need the OSRM answers
    if roads is None:
        notes.append("No OSRM answers yet: run `python -m src.research.osm_study fetch` on a PC with internet.")
    else:
        links = osm_link_table(graph, roads)
        links.to_csv(out / "osm_links.csv", index=False)
        notes.append(f"Real roads: {len(links)} links, detour factor median {links.detour.median():.2f}, "
                     f"mean {links.detour.mean():.2f}, max {links.detour.max():.2f}; "
                     f"OSM path follows our road on {links.follows_road.sum()}/{len(links)} links")

        real = osm_graph(graph, roads)
        peak = STUDY_DAY.replace(hour=8)
        cost_fns = (RouteFinder(graph, predictors[reference], cfg).edge_cost_fn(peak),
                    RouteFinder(real, predictors[reference], cfg).edge_cost_fn(peak))
        effect = route_effect(graph, real, cost_fns, site_pairs(graph), cfg["routing"]["speed_limit_kmh"])
        effect.to_csv(out / "route_effect.csv", index=False)
        changed = effect[effect.changed]
        notes.append(f"Route effect at 08:00 ({MODEL_LABELS[reference]}): {len(changed)}/{len(effect)} trips "
                     f"({100 * len(changed) / len(effect):.1f}%) change route; mean trip "
                     f"{effect.straight_min.mean():.2f} -> {effect.real_min.mean():.2f} min; time lost by the "
                     f"straight-line route: mean {changed.time_lost_s.mean() if len(changed) else 0:.1f} s, "
                     f"max {effect.time_lost_s.max():.1f} s")

    (out / "summary.txt").write_text("\n\n".join(notes) + "\n")
    print("\n\n".join(notes))
    print(f"\nResults written to {out}")


def main(argv: list[str]) -> None:
    """python -m src.research.osm_study fetch | run"""
    if argv[:1] == ["fetch"]:
        _, _, _, graph = _load_network()
        roads = fetch_osm_roads(graph, resolve_path(OUT_DIR) / OSM_ROADS_FILE)
        print(f"{len(roads)} links saved to {resolve_path(OUT_DIR) / OSM_ROADS_FILE}")
    elif argv[:1] == ["run"]:
        run()
    else:
        print(main.__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
