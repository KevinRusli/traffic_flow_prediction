"""End-to-end route guidance: (origin, destination, time, model) -> top-k routes.

OWNER: Kevin (Integration A2A). Uses Hassan's travel_time.py and the predictor.

This is what the GUI and `python main.py route ...` call. A Route carries
everything the GUI needs to look like a simple Google Maps result:
total minutes, distance, and the list of roads driven ("Warrigal Rd -> ...").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config import load_config
from src.graph.graph import Edge, RoadGraph
from src.prediction.predictor import FlowPredictor
from src.routing.travel_time import edge_travel_time
from src.search.top_k import k_shortest_paths


@dataclass
class Segment:
    """One edge of a route: drive from SCATS site `src` to `dst` along `road`."""
    src: int
    dst: int
    road: str
    distance_km: float
    time_s: float


@dataclass
class Route:
    path: list[int]                                  # SCATS sites, origin first
    segments: list[Segment] = field(default_factory=list)

    @property
    def travel_time_s(self) -> float:
        return sum(s.time_s for s in self.segments)

    @property
    def travel_time_min(self) -> float:
        return self.travel_time_s / 60.0

    @property
    def distance_km(self) -> float:
        return sum(s.distance_km for s in self.segments)

    @property
    def roads(self) -> list[str]:
        """Road names in driving order, consecutive duplicates merged.

        [Warrigal, Warrigal, Riversdale, Burke] -> [Warrigal, Riversdale, Burke]
        """
        names: list[str] = []
        for seg in self.segments:
            if seg.road and (not names or names[-1] != seg.road):
                names.append(seg.road)
        return names

    def __str__(self) -> str:
        via = " -> ".join(self.roads) or " -> ".join(str(s) for s in self.path)
        return f"{self.travel_time_min:.1f} min | {self.distance_km:.2f} km | via {via}"


class RouteFinder:
    def __init__(self, graph: RoadGraph, predictor: FlowPredictor, cfg: dict | None = None) -> None:
        self.graph = graph
        self.predictor = predictor
        self.cfg = cfg or load_config()

    def _edge_cost_fn(self, when: datetime):
        """Return cost_fn(edge) -> seconds, cached so each edge is predicted once."""
        use_dest = self.cfg["prediction"]["edge_flow_site"] == "destination"
        cache: dict[tuple[int, int], float] = {}

        def cost(edge: Edge) -> float:
            key = (edge.src, edge.dst)
            if key not in cache:
                site = edge.dst if use_dest else edge.src
                flow = self.predictor.predict_flow(site, when)
                cache[key] = edge_travel_time(edge.distance_km, flow, self.cfg)
            return cache[key]

        return cost

    def find_routes(
        self,
        origin: int,
        destination: int,
        when: datetime,
        k: int | None = None,
        method: str | None = None,
    ) -> list[Route]:
        """Return up to k routes sorted by estimated travel time (fastest first)."""
        for site in (origin, destination):
            if not self.graph.has_node(site):
                raise ValueError(f"SCATS site {site} is not in the Boroondara network")
        if origin == destination:
            return [Route([origin])]

        k = k or self.cfg["routing"]["k_paths"]
        method = method or self.cfg["routing"]["search_method"]
        cost_fn = self._edge_cost_fn(when)

        results = k_shortest_paths(self.graph, origin, destination, cost_fn, k, method,
                                   self.cfg["routing"]["speed_limit_kmh"])
        return [self._to_route(r.path, cost_fn) for r in results]

    def _to_route(self, path: list[int], cost_fn) -> Route:
        segments = []
        for a, b in zip(path, path[1:]):
            edge = next(e for e in self.graph.neighbours(a) if e.dst == b)
            segments.append(Segment(a, b, edge.road, edge.distance_km, cost_fn(edge)))
        return Route(path, segments)
