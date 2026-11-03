"""Regression metrics for comparing traffic-flow models.

OWNER: Sakib (evaluation, feeds the Research section)

All functions take UNscaled values (vehicles per 15 min) - call
TrafficDataset.inverse() first, otherwise the numbers are meaningless.

NRMSE follows the tutor's example: RMSE / mean(y_true).
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Sakib: implement mae()")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Sakib: implement rmse()")


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE divided by the mean of y_true (lower is better)."""
    raise NotImplementedError("Sakib: implement nrmse()")


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Mean absolute percentage error in %. Ignore/clip y_true near 0 (night-time counts)."""
    raise NotImplementedError("Sakib: implement mape()")


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Sakib: implement r2()")


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return every metric in one dict - used for the comparison table."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
