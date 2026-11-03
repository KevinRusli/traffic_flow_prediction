"""Turn the long SCATS table into ML-ready arrays.

OWNER: Kevin

Contract used by the model owners (Hassan, Ali Murtaza, Sakib):

    ds = build_dataset(df_long)     -> TrafficDataset
    ds.X_train  np.ndarray  shape (n_samples, lags)      scaled flows
    ds.y_train  np.ndarray  shape (n_samples,)           scaled next flow
    ds.X_test, ds.y_test             same, for the test period
    ds.meta_test pd.DataFrame        scats + location + timestamp of every test target
    ds.scaler                        fitted on TRAIN only; use ds.inverse(y) to get vehicles

Deep-learning models reshape X to (n, lags, 1) themselves - keep X 2-D here so
classic models (trees, SVR, ...) can use it directly.

A sample is the previous `lags` flows of one location (X) and the next flow (y).
Samples are assigned to train or test by the date of y, so the first test
samples use the last hours of training data as input. Windows never cross a
missing day.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import load_config


@dataclass
class TrafficDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Any
    meta_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta_test: pd.DataFrame = field(default_factory=pd.DataFrame)

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled flows back to vehicles per 15 minutes."""
        y_scaled = np.asarray(y_scaled).reshape(-1, 1)
        return self.scaler.inverse_transform(y_scaled).ravel()


def make_windows(series: np.ndarray, lags: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows over ONE continuous series.

    Example: series=[1,2,3,4,5], lags=3 -> X=[[1,2,3],[2,3,4]], y=[4,5]
    `horizon` = how many steps ahead y is (1 = the very next 15 minutes).
    A series too short for one window returns empty arrays.
    """
    series = np.asarray(series, dtype=float)
    n = len(series) - lags - horizon + 1
    if n <= 0:
        return np.empty((0, lags)), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(series, lags)[:n]
    y = series[lags + horizon - 1: lags + horizon - 1 + n]
    return X.copy(), y.copy()


def chronological_split(df_long: pd.DataFrame, train_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date (no shuffling - this is a time series).

    Rows on or before `train_end_date` go to train, later rows go to test.
    """
    in_train = df_long["timestamp"].dt.normalize() <= pd.Timestamp(train_end_date)
    return df_long[in_train], df_long[~in_train]


def _continuous_segments(group: pd.DataFrame, interval_minutes: int):
    """Yield runs of one location's rows with no missing 15-minute interval."""
    group = group.sort_values("timestamp")
    gap = group["timestamp"].diff() != pd.Timedelta(minutes=interval_minutes)
    for _, segment in group.groupby(gap.cumsum()):
        yield segment


def _make_scaler(kind: str):
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler '{kind}' (use minmax or standard)")


def build_dataset(df_long: pd.DataFrame, cfg: dict | None = None) -> TrafficDataset:
    """Scale flows, cut every continuous stretch into windows, split by target date."""
    cfg = cfg or load_config()
    dcfg = cfg["data"]
    lags, horizon = dcfg["lags"], dcfg.get("horizon", 1)
    minutes = dcfg.get("interval_minutes", 15)
    train_end = pd.Timestamp(dcfg["train_end_date"])

    # scaler sees training flows only
    train_rows, _ = chronological_split(df_long, dcfg["train_end_date"])
    scaler = _make_scaler(dcfg.get("scaler", "minmax"))
    scaler.fit(train_rows[["flow"]].to_numpy())

    X_parts, y_parts, meta_parts = [], [], []
    for location, group in df_long.groupby("location", sort=True):
        for segment in _continuous_segments(group, minutes):
            scaled = scaler.transform(segment[["flow"]].to_numpy()).ravel()
            X, y = make_windows(scaled, lags, horizon)
            if len(y) == 0:
                continue
            targets = segment.iloc[lags + horizon - 1: lags + horizon - 1 + len(y)]
            X_parts.append(X)
            y_parts.append(y)
            meta_parts.append(pd.DataFrame({
                "scats": targets["scats"].to_numpy(),
                "location": location,
                "timestamp": targets["timestamp"].to_numpy(),
            }))

    X = np.vstack(X_parts).astype("float32")
    y = np.concatenate(y_parts).astype("float32")
    meta = pd.concat(meta_parts, ignore_index=True)

    is_train = (meta["timestamp"].dt.normalize() <= train_end).to_numpy()
    return TrafficDataset(
        X_train=X[is_train], y_train=y[is_train],
        X_test=X[~is_train], y_test=y[~is_train],
        scaler=scaler,
        meta_train=meta[is_train].reset_index(drop=True),
        meta_test=meta[~is_train].reset_index(drop=True),
    )


def latest_window(df_long: pd.DataFrame, location: str, end_time: pd.Timestamp, lags: int) -> np.ndarray:
    """The `lags` raw flows of `location` just before `end_time`.

    The data only covers October 2006. If the window is not in the data, the same
    weekday and time of day on the latest available date is used (any Monday
    08:00 -> Mon 30 Oct 2006 08:00), then the same time on any date.
    """
    series = (df_long[df_long["location"] == location]
              .set_index("timestamp")["flow"].sort_index())
    if series.empty:
        raise ValueError(f"Unknown location: {location!r}")

    step = pd.Timedelta(minutes=15)
    end_time = pd.Timestamp(end_time).floor("15min")
    time_of_day = end_time - end_time.normalize()

    def window_at(t: pd.Timestamp):
        wanted = pd.date_range(t - lags * step, periods=lags, freq=step)
        values = series.reindex(wanted)
        return None if values.isna().any() else values.to_numpy()

    days = sorted(series.index.normalize().unique(), reverse=True)
    same_weekday = [d for d in days if d.dayofweek == end_time.dayofweek]
    for t in [end_time] + [d + time_of_day for d in same_weekday + days]:
        window = window_at(t)
        if window is not None:
            return window
    raise ValueError(f"No complete {lags}-step window for {location!r} at {end_time:%H:%M}")
