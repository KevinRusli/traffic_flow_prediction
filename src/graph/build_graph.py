"""Build the Boroondara road graph from the SCATS location names.

OWNER: Kevin

The dataset has no list of road links, so they are inferred from the location
names. "WARRIGAL_RD N of TOORAK_RD" at site 2000 means site 2000 lies on both
Warrigal Rd and Toorak Rd. For every road, its sites are sorted along the road
and each site is linked to the next one, in both directions.

Corrections, all checked against docs/Boroondara_SCATS_map.pdf:
  - one road under two names: HARP_ST = HARP_RD (site 4321 uses both),
    MAROONDAH_HWY = WHITEHORSE_RD, VICTORIA_ST = BARKERS_RD, BURWOOD_HWY = TOORAK_RD,
    and the two Eastern Fwy ramp names = EASTERN_FWY
  - one name for two roads: HIGH_ST at site 2846 is High St in Glen Iris,
    not High St in Kew where the other HIGH_ST sites are
  - a site that shares no road with any other site (4812, 2846) is linked to
    its nearest site, so every site can be routed to

Edge length is the straight-line distance between the two sites.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from src.graph.graph import Node, RoadGraph

_LOCATION_RE = re.compile(r"^(?P<road>.+?) (?P<dir>NE|NW|SE|SW|N|S|E|W) of (?P<cross>.+?)(?: #\d+)?$",
                          re.IGNORECASE)

ROAD_ALIASES = {
    "HARP_ST": "HARP_RD",
    "MAROONDAH_HWY": "WHITEHORSE_RD",
    "VICTORIA_ST": "BARKERS_RD",
    "BURWOOD_HWY": "TOORAK_RD",
    "EASTERN_FWY_W_BD_RAMPS": "EASTERN_FWY",
    "OFFRAMP_EASTERN_FWY": "EASTERN_FWY",
}
# (site, road) pairs that share a name with a different road elsewhere
SITE_ROAD_OVERRIDES = {(2846, "HIGH_ST"): "HIGH_ST_GLEN_IRIS"}
DISPLAY_NAMES = {"S.E.ARTERIAL": "South Eastern Arterial", "HIGH_ST_GLEN_IRIS": "High St"}


def parse_location(location: str) -> tuple[str, str, str]:
    """Split "WARRIGAL_RD N of TOORAK_RD" -> ("WARRIGAL_RD", "N", "TOORAK_RD").

    "of"/"OF" both work and a detector suffix like " #2" is ignored.
    """
    match = _LOCATION_RE.match(location.strip())
    if match is None:
        raise ValueError(f"Unrecognised location format: {location!r}")
    return match["road"].strip(), match["dir"].upper(), match["cross"].strip()


def display_name(road: str) -> str:
    """"WARRIGAL_RD" -> "Warrigal Rd", "HIGH STREET_RD" -> "High Street Rd"."""
    if road in DISPLAY_NAMES:
        return DISPLAY_NAMES[road]
    return " ".join(word.capitalize() for word in road.replace("_", " ").split())


def _canonical_road(site: int, road: str) -> str:
    road = ROAD_ALIASES.get(road, road)
    return SITE_ROAD_OVERRIDES.get((site, road), road)


def site_roads(df_long: pd.DataFrame) -> dict[int, list[str]]:
    """Roads each site lies on, most frequently named first."""
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (site, location), _ in df_long.groupby(["scats", "location"]):
        road, _, cross = parse_location(location)
        for r in (road, cross):
            counts[site][_canonical_road(site, r)] += 1
    return {site: sorted(c, key=lambda r: (-c[r], r)) for site, c in counts.items()}


def _order_along_road(graph: RoadGraph, sites: list[int]) -> list[int]:
    """Sort sites by their position along the road's main direction."""
    lat0 = math.radians(np.mean([graph.nodes[s].lat for s in sites]))
    xy = np.array([[graph.nodes[s].lon * math.cos(lat0), graph.nodes[s].lat] for s in sites])
    centred = xy - xy.mean(axis=0)
    direction = np.linalg.svd(centred)[2][0]
    return [sites[i] for i in np.argsort(centred @ direction)]


def build_graph(df_long: pd.DataFrame) -> RoadGraph:
    """Return the RoadGraph: one node per SCATS site, two-way edges along shared roads."""
    graph = RoadGraph()
    roads_of = site_roads(df_long)
    coords = df_long.groupby("scats")[["lat", "lon"]].median()
    locations = df_long.groupby("scats")["location"].unique()

    for site in sorted(roads_of):
        name = " / ".join(display_name(r) for r in roads_of[site][:2])
        graph.add_node(Node(int(site), float(coords.at[site, "lat"]), float(coords.at[site, "lon"]),
                            name, sorted(locations[site])))

    sites_on: dict[str, list[int]] = defaultdict(list)
    for site, roads in roads_of.items():
        for road in roads:
            sites_on[road].append(site)

    for road, sites in sorted(sites_on.items()):
        if len(sites) < 2:
            continue
        ordered = _order_along_road(graph, sorted(sites))
        for a, b in zip(ordered, ordered[1:]):
            _add_two_way(graph, a, b, display_name(road))

    for site in sorted(graph.nodes):
        if not graph.neighbours(site):
            nearest = min((s for s in graph.nodes if s != site), key=lambda s: graph.haversine_km(site, s))
            _add_two_way(graph, site, nearest, display_name(roads_of[site][0]))
    return graph


def _add_two_way(graph: RoadGraph, a: int, b: int, road: str) -> None:
    graph.add_edge(a, b, road=road)
    graph.add_edge(b, a, road=road)
