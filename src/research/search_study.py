"""Research: our 2A search methods and the ML predictions on the real road network.

OWNER: Kevin (Research initiative, shared with Sakib)

Run: python main.py research

Every ordered pair of SCATS sites (40 x 39 = 1,560 trips) is routed, and two
questions are answered:

1. Search methods. Each 2A method finds one route (k = 1) with predicted travel
   times. The fastest possible time comes from a separate Dijkstra search, so
   A* is checked too: we record how often each method finds the fastest route,
   how much slower its route is, how many nodes it creates and how long it takes. Done at a peak (08:00) and a quiet
   time (03:00), because the edge costs change with traffic.
2. Value of the prediction. A* is run with different edge costs: free flow
   (no traffic data, 60 km/h everywhere), each ML model at 08:00, and the
   reference model at 03:00. For each trip we check whether the fastest route
   differs from the reference model's 08:00 route, and how much time a driver
   loses at 08:00 by following that other route instead.

Outputs: results/research/methods.csv, route_choice.csv and search_methods.png.
"""
from __future__ import annotations

import heapq
from datetime import datetime
from itertools import permutations
from pathlib import Path
from time import perf_counter

import matplotlib
import pandas as pd

from src.config import load_config, resolve_path
from src.graph.graph import RoadGraph
from src.prediction.predictor import ConstantPredictor, ModelPredictor
from src.routing.route_finder import RouteFinder
from src.search.adapter import CostFn, TravelTimeProblem
from src.search.methods import SEARCH_METHODS
from src.search.top_k import k_shortest_paths

matplotlib.use("Agg")                                # write PNG files, no window
from matplotlib import pyplot as plt  # noqa: E402

PEAK = datetime(2006, 10, 30, 8, 0)    # Monday of the test week, morning peak
NIGHT = datetime(2006, 10, 30, 3, 0)   # same Monday, quietest hour
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "third": "Random Forest"}


def site_pairs(graph: RoadGraph) -> list[tuple[int, int]]:
    """Every (origin, destination) pair of different sites."""
    return list(permutations(sorted(graph.nodes), 2))


def path_time(graph: RoadGraph, path: list[int], cost_fn: CostFn) -> float:
    """Seconds needed to drive `path` when the edges cost cost_fn(edge)."""
    return TravelTimeProblem(graph, path[0], [path[-1]], cost_fn).path_cost(path)


def fastest_time(graph: RoadGraph, origin: int, goal: int, cost_fn: CostFn) -> float | None:
    """Fastest travel time by Dijkstra's algorithm, or None if the goal is unreachable.

    Written apart from the 2A code so that it is an independent check of A*.
    """
    done = set()
    frontier = [(0.0, origin)]                     # (seconds from the origin, site)
    while frontier:
        seconds, site = heapq.heappop(frontier)
        if site == goal:
            return seconds
        if site in done:                           # already reached by a faster path
            continue
        done.add(site)
        for edge in graph.neighbours(site):
            if edge.dst not in done:
                heapq.heappush(frontier, (seconds + cost_fn(edge), edge.dst))
    return None


def compare_methods(graph: RoadGraph, cost_fn: CostFn, pairs, speed_limit_kmh: float = 60.0) -> pd.DataFrame:
    """One row per (trip, method): route time, fastest time, nodes created, run time."""
    rows = []
    for origin, goal in pairs:
        best = fastest_time(graph, origin, goal, cost_fn)
        if best is None:                              # goal cannot be reached
            continue
        found = {}
        for method in SEARCH_METHODS:
            start = perf_counter()
            result = k_shortest_paths(graph, origin, goal, cost_fn, 1, method, speed_limit_kmh)
            ms = 1000 * (perf_counter() - start)
            found[method] = (result[0], ms)
        for method, (route, ms) in found.items():
            rows.append(dict(origin=origin, goal=goal, method=method, time_s=route.cost,
                             best_s=best, nodes=route.nodes_created, edges=len(route.path) - 1, ms=ms))
    df = pd.DataFrame(rows)
    df["extra_pct"] = 100 * (df.time_s - df.best_s) / df.best_s
    return df


def summarise_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Per method: share of trips solved optimally, extra time, nodes and run time."""
    df = df.assign(optimal=df.extra_pct < 1e-6)        # equal to the fastest time up to rounding
    table = df.groupby("method", sort=False).agg(
        optimal_pct=("optimal", "mean"), mean_extra_pct=("extra_pct", "mean"),
        max_extra_pct=("extra_pct", "max"), mean_edges=("edges", "mean"),
        mean_nodes=("nodes", "mean"), mean_ms=("ms", "mean"))
    table["optimal_pct"] *= 100
    return table


def compare_route_choice(graph: RoadGraph, cost_fns: dict[str, CostFn], reference: str,
                         pairs, speed_limit_kmh: float = 60.0) -> pd.DataFrame:
    """Per scenario: its mean trip, how often its fastest route differs from the
    reference one, and the time lost when that route is driven in the reference traffic."""
    best = {name: {} for name in cost_fns}             # scenario -> trip -> fastest SearchResult
    for name, cost_fn in cost_fns.items():
        for origin, goal in pairs:
            routes = k_shortest_paths(graph, origin, goal, cost_fn, 1, "AS", speed_limit_kmh)
            if routes:
                best[name][(origin, goal)] = routes[0]

    ref_fn, ref_routes = cost_fns[reference], best[reference]
    rows = []
    for name, routes in best.items():
        lost = []                                       # seconds lost on trips whose route differs
        for trip, route in routes.items():
            ref_path = ref_routes[trip].path
            if route.path != ref_path:
                lost.append(path_time(graph, route.path, ref_fn) - path_time(graph, ref_path, ref_fn))
        lost = pd.Series(lost, dtype=float)
        paths = [r.path for r in routes.values()]
        rows.append(dict(scenario=name, trips=len(paths),
                         mean_trip_s=sum(r.cost for r in routes.values()) / len(paths),
                         mean_edges=sum(len(p) - 1 for p in paths) / len(paths),
                         changed_pct=100 * len(lost) / len(paths),
                         mean_lost_s=lost.mean() if len(lost) else 0.0,
                         max_lost_s=lost.max() if len(lost) else 0.0))
    return pd.DataFrame(rows).set_index("scenario")


def _plot_methods(table: pd.DataFrame, path: Path) -> None:
    """Two panels for the 08:00 run: share of trips solved optimally, and mean nodes created."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, column, title in [(axes[0], "optimal_pct", "Trips where the fastest route is found (%)"),
                              (axes[1], "mean_nodes", "Nodes created per search (mean)")]:
        ax.bar(table.index, table[column], width=0.6)
        for x, value in enumerate(table[column]):
            ax.text(x, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, table[column].max() * 1.12)       # room for the labels above the bars
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_study(graph: RoadGraph, cfg: dict | None = None, model: str = "third") -> dict[str, pd.DataFrame]:
    """Run both experiments with `model` as the reference and save CSVs and figures."""
    cfg = cfg or load_config()
    limit = cfg["routing"]["speed_limit_kmh"]
    out = resolve_path(cfg["paths"]["results_dir"]) / "research"
    out.mkdir(parents=True, exist_ok=True)
    pairs = site_pairs(graph)

    # 1. search methods, with the reference model's predicted travel times
    finder = RouteFinder(graph, ModelPredictor(model, cfg), cfg)
    peak = summarise_methods(compare_methods(graph, finder.edge_cost_fn(PEAK), pairs, limit))
    night = summarise_methods(compare_methods(graph, finder.edge_cost_fn(NIGHT), pairs, limit))
    methods = pd.concat({"08:00": peak, "03:00": night}, names=["time", "method"])

    # 2. route choice: free flow, every trained model at 08:00, reference model at 03:00
    reference = f"{MODEL_LABELS[model]} 08:00"
    free_flow = RouteFinder(graph, ConstantPredictor(0.0), cfg)    # flow 0 -> 60 km/h everywhere
    cost_fns = {reference: finder.edge_cost_fn(PEAK),
                "Free flow (no prediction)": free_flow.edge_cost_fn(PEAK),
                f"{MODEL_LABELS[model]} 03:00": finder.edge_cost_fn(NIGHT)}
    for other in cfg["models"]["available"]:
        if other == model:
            continue
        try:
            predictor = ModelPredictor(other, cfg)
        except (FileNotFoundError, ValueError) as exc:  # Keras raises ValueError for a missing file
            print(f"[research] {other} skipped, not trained: {exc}")
            continue
        cost_fns[f"{MODEL_LABELS[other]} 08:00"] = RouteFinder(graph, predictor, cfg).edge_cost_fn(PEAK)
    choice = compare_route_choice(graph, cost_fns, reference, pairs, limit)

    methods.round(3).to_csv(out / "methods.csv")
    choice.round(3).to_csv(out / "route_choice.csv")
    _plot_methods(peak, out / "search_methods.png")
    return dict(methods=methods, route_choice=choice)
