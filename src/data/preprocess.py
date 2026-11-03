"""Turn the long SCATS table into ML-ready arrays.

OWNER: Kevin (Data processing)

Contract used by the model owners (Hassan, Ali Murtaza, Sakib):

    ds = build_dataset(df_long)     -> TrafficDataset
    ds.X_train  np.ndarray  shape (n_samples, lags)      scaled flows
    ds.y_train  np.ndarray  shape (n_samples,)           scaled next flow
    ds.X_test, ds.y_test             same, for the test period
    ds.meta_test pd.DataFrame        location + timestamp of every test sample
    ds.scaler                        fitted on TRAIN only; use ds.inverse(y) to get vehicles

Deep-learning models reshape X to (n, lags, 1) themselves - keep X 2-D here so
classic models (trees, SVR, ...) can use it directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

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

    TODO(Kevin): implement. Do NOT build windows across a gap in the data
    (missing days) or across two different locations - split the series first.
    """
    raise NotImplementedError("Kevin: implement make_windows()")


def chronological_split(df_long: pd.DataFrame, train_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date (no shuffling - this is a time series).

    TODO(Kevin): rows with timestamp.date() <= train_end_date go to train.
    """
    raise NotImplementedError("Kevin: implement chronological_split()")


def build_dataset(df_long: pd.DataFrame, cfg: dict | None = None) -> TrafficDataset:
    """Full pipeline: split -> fit scaler on train -> windows per location -> stack.

    TODO(Kevin): implement using the helpers above and cfg["data"].
    Optional extras for the Research section: time-of-day / day-of-week features,
    one-hot location, neighbouring-site flows.
    """
    cfg = cfg or load_config()
    raise NotImplementedError("Kevin: implement build_dataset()")


def latest_window(df_long: pd.DataFrame, location: str, end_time: pd.Timestamp, lags: int) -> np.ndarray:
    """Return the `lags` raw flows of `location` that end just before `end_time`.

    Used by the predictor at query time. If `end_time` is outside October 2006,
    map it to the same weekday + time-of-day inside the dataset (document this
    assumption in the report).

    TODO(Kevin)
    """
    raise NotImplementedError("Kevin: implement latest_window()")
