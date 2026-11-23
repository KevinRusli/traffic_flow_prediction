"""Regression metrics for comparing traffic-flow models.

OWNER: Sakib (evaluation, feeds the Research section)

All functions take UNscaled values (vehicles per 15 min) - call
TrafficDataset.inverse() first, otherwise the numbers are meaningless.

NRMSE follows the tutor's example: RMSE / mean(y_true).
MAPE is a percentage with its denominator clipped to eps (default: 1 vehicle).
NRMSE with zero mean and R2 with fewer than two samples return NaN.
"""
from __future__ import annotations

import numpy as np


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require matching, non-empty 1-D arrays containing finite numbers."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    # Reject column vectors rather than accidentally broadcasting them into a matrix.
    if actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must be matching 1-D arrays.")
    if actual.size == 0:
        raise ValueError("At least one actual/predicted pair is required.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must contain only finite numbers.")
    return actual, predicted


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average absolute error, in vehicles per 15 minutes (lower is better)."""
    actual, predicted = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error; larger errors receive more weight."""
    actual, predicted = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE / mean actual flow, as a ratio; NaN if the mean is zero."""
    actual, predicted = _validate_inputs(y_true, y_pred)
    mean_flow = float(np.mean(actual))
    if mean_flow == 0:
        return float("nan")  # Normalisation is undefined when there is no mean flow.
    return rmse(actual, predicted) / mean_flow


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Percentage error using max(abs(actual), eps); includes zero-count samples."""
    actual, predicted = _validate_inputs(y_true, y_pred)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number.")
    # A one-vehicle floor prevents division by zero for night-time traffic counts.
    denominator = np.maximum(np.abs(actual), eps)
    return float(100 * np.mean(np.abs(actual - predicted) / denominator))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination: best is 1; scores can be negative.

    For constant actual values, use 1 for perfect predictions and 0 otherwise,
    matching scikit-learn's default convention. A single sample returns NaN.
    """
    actual, predicted = _validate_inputs(y_true, y_pred)
    if actual.size < 2:
        return float("nan")
    residual_sum = float(np.sum((actual - predicted) ** 2))
    total_sum = float(np.sum((actual - np.mean(actual)) ** 2))
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0
    return 1.0 - residual_sum / total_sum


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return every metric in one dict - used for the comparison table."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
