"""Build the Boroondara RoadGraph from the SCATS dataset.

OWNER: Kevin (Data processing)

The dataset does NOT give you the road links directly. You have to infer them:

  Each `location` looks like  "<ROAD_A> <DIR> of <ROAD_B>", e.g.
      site 2000: "WARRIGAL_RD N of TOORAK_RD"
  meaning: detector on WARRIGAL_RD, on the north approach of the TOORAK_RD
  intersection. So site 2000 sits on WARRIGAL_RD and TOORAK_RD.

Suggested approach (describe your final approach in the report):
  1. For every site, collect the set of roads it sits on (normalise names:
     upper-case, "HIGH STREET_RD" vs "HIGH_ST" etc.).
  2. Two sites are linked if they share a road AND they are the closest sites
     to each other along that road in the given direction (N/S/E/W of ...).
  3. Add edges in both directions (roads are two-way) with haversine distance,
     and ALWAYS pass the road name: graph.add_edge(a, b, road="Warrigal Rd").
     The GUI shows these names ("Warrigal Rd -> Riversdale Rd -> ..."), so make
     them readable: "WARRIGAL_RD" -> "Warrigal Rd", "HIGH STREET_RD" -> "High Street Rd".
  4. Sanity-check against docs/Boroondara_SCATS_map.pdf and plot it
     (notebooks/01_graph_check.ipynb) - e.g. 2000 -> 3002 must be reachable.

Alternatively hand-write the adjacency list in data/processed/edges.csv
(src,dst,road) after checking the map, and just load it here - totally fine
as long as you explain it.
"""
from __future__ import annotations

import pandas as pd

from src.graph.graph import RoadGraph


def parse_location(location: str) -> tuple[str, str, str]:
    """Split "WARRIGAL_RD N of TOORAK_RD" -> ("WARRIGAL_RD", "N", "TOORAK_RD").

    TODO(Kevin): handle "OF" / "of", directions like NE/SW, and names with spaces.
    """
    raise NotImplementedError("Kevin: implement parse_location()")


def build_graph(df_long: pd.DataFrame) -> RoadGraph:
    """Return the RoadGraph with one node per SCATS site and inferred edges.

    TODO(Kevin)
    """
    raise NotImplementedError("Kevin: implement build_graph()")
