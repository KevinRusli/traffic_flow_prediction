"""Load the raw SCATS October 2006 dataset into clean pandas structures.

OWNER: Kevin

Contract (other members depend on these exact column names - do not rename):

    load_raw(path)  -> wide DataFrame, one row per (location, day), columns V00..V95
    to_long(df_wide) -> long DataFrame with columns:
        scats     int              SCATS site number, e.g. 2000
        location  str              approach description, e.g. "WARRIGAL_RD N of TOORAK_RD"
        lat, lon  float            coordinates (offset-corrected, 0.0 values fixed)
        timestamp pandas.Timestamp start of the 15-min interval
        flow      float            vehicles counted in that 15-min interval

Data quirks handled here (details in docs/DATA_NOTES.md): the header is on the
2nd row, Date says 00:15 for a day starting at 00:00, site 4335 has two
detectors under one name, one approach of site 4266 has no coordinates, and all
coordinates are offset from the real intersections.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path

VOLUME_COLUMNS = [f"V{i:02d}" for i in range(96)]
LONG_COLUMNS = ["scats", "location", "lat", "lon", "timestamp", "flow"]

_RENAME = {
    "SCATS Number": "scats",
    "Location": "location",
    "NB_LATITUDE": "lat",
    "NB_LONGITUDE": "lon",
    "Date": "date",
}
_DETECTOR_ID = "VR Internal Loc"


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the 'Data' sheet and return the wide table (one row per location per day)."""
    cfg = load_config()
    path = resolve_path(path or cfg["paths"]["scats_data"])
    raw = pd.read_excel(path, sheet_name=cfg["data"]["sheet_name"], header=cfg["data"]["header_row"])

    df = raw[list(_RENAME) + [_DETECTOR_ID] + VOLUME_COLUMNS].rename(columns=_RENAME)
    df["scats"] = df["scats"].astype(int)
    df["location"] = df["location"].str.strip()
    # Date is stored as 'YYYY-MM-DD 00:15' but V00 starts at 00:00
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df[VOLUME_COLUMNS] = df[VOLUME_COLUMNS].astype(float)

    df = _split_shared_location_names(df)
    df = df.drop(columns=[_DETECTOR_ID]).drop_duplicates()
    return df.sort_values(["location", "date"]).reset_index(drop=True)


def _split_shared_location_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the 2nd detector sharing a location name to "<name> #2".

    Site 4335 has two detectors (different VR Internal Loc and counts) both
    called "HIGH_ST NE of CHARLES_ST"; merging them would mix two series.
    """
    detectors = df.groupby("location")[_DETECTOR_ID].unique()
    for location, ids in detectors.items():
        for n, det in enumerate(sorted(ids)[1:], start=2):
            rows = (df["location"] == location) & (df[_DETECTOR_ID] == det)
            df.loc[rows, "location"] = f"{location} #{n}"
    return df


def to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Melt V00..V95 into one row per 15-minute interval (see LONG_COLUMNS)."""
    minutes = load_config()["data"]["interval_minutes"]
    long = df_wide.melt(
        id_vars=["scats", "location", "lat", "lon", "date"],
        value_vars=VOLUME_COLUMNS,
        var_name="interval",
        value_name="flow",
    )
    # "V37" -> 37 * 15 minutes after midnight
    offset = pd.to_timedelta(long["interval"].str[1:].astype(int) * minutes, unit="min")
    long["timestamp"] = long["date"] + offset
    long = long.sort_values(["location", "timestamp"]).reset_index(drop=True)
    return long[LONG_COLUMNS]


def fix_coordinates(df: pd.DataFrame, counts_csv: str | Path | None = None) -> pd.DataFrame:
    """Fill missing (0.0) coordinates, then shift everything by the config offset.

    A missing approach takes the median of the other approaches of the same site.
    Only if a whole site is missing is it looked up in the Traffic_Count CSV, whose
    coordinates are already real-world, so they skip the offset. The offset was
    checked against that CSV on sites 2000, 3002, 970 and 4266
    (difference ~ +0.0015 lat, +0.0013 lon).
    """
    cfg = load_config()
    off = cfg["data"]["coord_offset"]
    df = df.copy()

    missing = (df["lat"] == 0) | (df["lon"] == 0) | df["lat"].isna() | df["lon"].isna()
    df.loc[~missing, "lat"] += off["lat"]
    df.loc[~missing, "lon"] += off["lon"]

    counts = None
    for site in df.loc[missing, "scats"].unique():
        rows = missing & (df["scats"] == site)
        known = ~missing & (df["scats"] == site)
        if known.any():
            # approaches of one site are within ~50 m of each other
            lat, lon = df.loc[known, "lat"].median(), df.loc[known, "lon"].median()
        else:
            if counts is None:
                csv_path = resolve_path(counts_csv or "data/raw/Traffic_Count_Locations_with_LONG_LAT.csv")
                counts = pd.read_csv(csv_path, usecols=["X", "Y", "SITE_DESC"])
            lat, lon = _lookup_site(df[df["scats"] == site]["location"].unique(), counts)
        df.loc[rows, "lat"], df.loc[rows, "lon"] = lat, lon
    return df


def _lookup_site(locations, counts: pd.DataFrame) -> tuple[float, float]:
    """Median real-world (lat, lon) of count points whose description names both roads."""
    from src.graph.build_graph import parse_location

    road, _, cross = parse_location(locations[0])
    a, b = (r.replace("_", " ").upper() for r in (road, cross))
    desc = counts["SITE_DESC"].fillna("").str.upper()
    hits = counts[desc.str.contains(a, regex=False) & desc.str.contains(b, regex=False)]
    if hits.empty:
        raise ValueError(f"No coordinates found for {locations[0]} - add them by hand")
    return float(np.median(hits["Y"])), float(np.median(hits["X"]))


def load_long(path: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Cleaned long DataFrame, cached in data/processed/ after the first call.

    Delete the cache file after changing this module.
    """
    cfg = load_config()
    cache = resolve_path(cfg["paths"]["processed_dir"]) / "scats_long.csv"
    if use_cache and cache.exists():
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        return df[LONG_COLUMNS]

    df = fix_coordinates(to_long(load_raw(path)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df[LONG_COLUMNS]
