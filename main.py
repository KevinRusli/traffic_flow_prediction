"""TBRGS command-line entry point.

    python main.py train    --model lstm|gru|third|all
    python main.py evaluate
    python main.py route    --origin 2000 --dest 3002 --time "2006-10-15 08:00" [--model gru] [--k 5] [--method AS]
    python main.py gui
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config

ROOT = Path(__file__).resolve().parent


def cmd_train(args: argparse.Namespace) -> None:
    from src.training.train import train_all, train_model

    cfg = load_config(args.config)
    if args.model == "all":
        train_all(cfg)
    else:
        train_model(args.model, cfg)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.evaluation.compare import evaluate_all

    table = evaluate_all(load_config(args.config))
    print(table.to_string())


def cmd_route(args: argparse.Namespace) -> None:
    from src.data.loader import load_long
    from src.graph.build_graph import build_graph
    from src.prediction.predictor import ConstantPredictor, ModelPredictor
    from src.routing.route_finder import RouteFinder

    cfg = load_config(args.config)
    graph = build_graph(load_long())
    predictor = ConstantPredictor() if args.model == "constant" else ModelPredictor(args.model, cfg)
    finder = RouteFinder(graph, predictor, cfg)
    when = datetime.strptime(args.time, "%Y-%m-%d %H:%M")

    routes = finder.find_routes(args.origin, args.dest, when, k=args.k, method=args.method)
    if not routes:
        print(f"No route found from {args.origin} to {args.dest}.")
        return
    print(f"Top {len(routes)} route(s) from {args.origin} to {args.dest} at {when:%Y-%m-%d %H:%M} "
          f"[model={args.model}, search={args.method}]")
    for i, route in enumerate(routes, 1):
        print(f"  {i}. {route}")
        print(f"     sites: {' -> '.join(str(s) for s in route.path)}")


def cmd_gui(_: argparse.Namespace) -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "gui.py")], check=False)


def build_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Traffic-Based Route Guidance System (COS30019 A2B)")
    p.add_argument("--config", default=None, help="path to an alternative config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train ML model(s)")
    t.add_argument("--model", default="all", choices=cfg["models"]["available"] + ["all"])
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="compare all trained models")
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("route", help="find top-k routes between two SCATS sites")
    r.add_argument("--origin", type=int, default=cfg["gui"]["default_origin"])
    r.add_argument("--dest", type=int, default=cfg["gui"]["default_destination"])
    r.add_argument("--time", default=cfg["gui"]["default_datetime"], help='"YYYY-MM-DD HH:MM"')
    r.add_argument("--model", default=cfg["models"]["default"],
                   choices=cfg["models"]["available"] + ["constant"])
    r.add_argument("--k", type=int, default=cfg["routing"]["k_paths"])
    r.add_argument("--method", default=cfg["routing"]["search_method"],
                   choices=["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"], type=str.upper)
    r.set_defaults(func=cmd_route)

    g = sub.add_parser("gui", help="launch the Streamlit GUI")
    g.set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
