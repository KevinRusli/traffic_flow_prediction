"""Load the raw SCATS October 2006 dataset into clean pandas structures.

OWNER: Kevin (Data processing)

Contract (other members depend on these exact column names - do not rename):

    load_raw(path)  -> wide DataFrame, one row per (location, day), columns V00..V95
    to_long(df_wide) -> long DataFrame with columns:
        scats     int              SCATS site number, e.g. 2000
        location  str              approach description, e.g. "WARRIGAL_RD N of TOORAK_RD"
        lat, lon  float            coordinates (offset-corrected, 0.0 values fixed)
        timestamp pandas.Timestamp start of the 15-min interval
        flow      float            vehicles counted in that 15-min interval

See docs/DATA_NOTES.md for the quirks you have to handle (header row, duplicate
rows, zero coordinates, missing days, the odd 00:15 timestamp in the Date column).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

VOLUME_COLUMNS = [f"V{i:02d}" for i in range(96)]
LONG_COLUMNS = ["scats", "location", "lat", "lon", "timestamp", "flow"]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the 'Data' sheet of the SCATS workbook and return the wide table.

    TODO(Kevin):
      - read with the correct header row (config: data.header_row)
      - keep only: SCATS Number, Location, NB_LATITUDE, NB_LONGITUDE, Date, V00..V95
      - rename SCATS Number -> scats, Location -> location, NB_LATITUDE -> lat,
        NB_LONGITUDE -> lon, Date -> date
      - normalise `date` to midnight (the raw value is 'YYYY-MM-DD 00:15')
      - drop exact duplicate rows (site 4335 is duplicated)
    """
    cfg = load_config()
    path = resolve_path(path or cfg["paths"]["scats_data"])
    raise NotImplementedError("Kevin: implement load_raw()")


def to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Melt V00..V95 into one row per 15-minute interval (see LONG_COLUMNS).

    TODO(Kevin): timestamp = date + interval_index * 15 minutes, sorted by
    (location, timestamp).
    """
    raise NotImplementedError("Kevin: implement to_long()")


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Fix zero lat/lon (site 4266) and apply the offset from config.

    TODO(Kevin): fill missing coordinates from the site listing / Traffic_Count CSV
    or from the other rows of the same site, then add data.coord_offset.
    """
    raise NotImplementedError("Kevin: implement fix_coordinates()")


def load_long(path: str | Path | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Convenience entry point used by everyone else.

    Returns the cleaned long DataFrame. Caches it as a parquet/csv file in
    data/processed/ so the Excel file is only parsed once.
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
