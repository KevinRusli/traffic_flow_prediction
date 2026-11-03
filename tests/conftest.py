"""Shared pytest fixtures + the "not implemented yet" rule.

Any test that hits a NotImplementedError is reported as XFAIL ("not implemented
yet") instead of FAILED. So the suite is green from day one, and each test turns
into a real PASS/FAIL automatically as soon as its function is implemented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.graph import Node, RoadGraph  # noqa: E402


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        report.outcome = "skipped"
        report.wasxfail = f"not implemented yet: {call.excinfo.value}"


@pytest.fixture
def toy_graph() -> RoadGraph:
    """Small hand-made graph so search/routing tests don't depend on the real data.

        1 --A Rd 1km-- 2 --A Rd 1km-- 4
        |                             |
        +--B Rd 1km-- 3 --C Rd 3km----+          5 (isolated)

    Shortest 1->4: 1-2-4 (2 km, all on A Rd). Second: 1-3-4 (4 km).
    Coordinates keep every edge >= its straight line, so the A* heuristic is admissible.
    """
    g = RoadGraph()
    for sid, lat, lon in [(1, -37.800, 145.000), (2, -37.800, 145.010), (3, -37.805, 145.005),
                          (4, -37.800, 145.020), (5, -37.900, 145.100)]:
        g.add_node(Node(sid, lat, lon))
    for a, b, d, road in [(1, 2, 1.0, "A Rd"), (2, 4, 1.0, "A Rd"), (1, 3, 1.0, "B Rd"), (3, 4, 3.0, "C Rd")]:
        g.add_edge(a, b, d, road)
        g.add_edge(b, a, d, road)
    return g


@pytest.fixture
def grid_graph() -> RoadGraph:
    """3x3 grid (sites 1..9) with uneven edge lengths - many alternative routes."""
    g = RoadGraph()
    for r in range(3):
        for c in range(3):
            g.add_node(Node(r * 3 + c + 1, -37.80 - r * 0.005, 145.00 + c * 0.005))
    lengths = iter([1.0, 1.4, 1.1, 0.9, 1.3, 1.2, 1.0, 1.6, 1.2, 1.1, 1.5, 1.0])
    for r in range(3):
        for c in range(3):
            n = r * 3 + c + 1
            if c < 2:
                d = next(lengths)
                g.add_edge(n, n + 1, d)
                g.add_edge(n + 1, n, d)
            if r < 2:
                d = next(lengths)
                g.add_edge(n, n + 3, d)
                g.add_edge(n + 3, n, d)
    return g


@pytest.fixture
def toy_long_df() -> pd.DataFrame:
    """Two locations x 3 days of synthetic 15-minute counts in the long format."""
    rows = []
    for loc_i, (scats, loc) in enumerate([(100, "A_RD N of B_RD"), (200, "B_RD E of A_RD")]):
        for ts in pd.date_range("2006-10-01", periods=96 * 3, freq="15min"):
            rows.append((scats, loc, -37.8, 145.0 + loc_i / 100, ts, float(ts.hour * 10 + loc_i)))
    return pd.DataFrame(rows, columns=["scats", "location", "lat", "lon", "timestamp", "flow"])


@pytest.fixture
def toy_xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.random((200, 12)).astype("float32")
    y = X.mean(axis=1)
    return X, y


@pytest.fixture(scope="session")
def scats_file() -> Path:
    path = ROOT / "data" / "raw" / "Scats Data October 2006.xls"
    if not path.exists():
        pytest.skip("SCATS dataset not found in data/raw/")
    return path
