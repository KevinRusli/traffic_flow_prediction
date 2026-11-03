"""Compare all trained models on the same test set and produce report material.

OWNER: Sakib (evaluation, feeds the Research section)

    python main.py evaluate

Outputs (all go to results/, used directly in the report):
  results/metrics.csv                 one row per model, columns = metrics
  results/metrics_by_site.csv         per-location RMSE (which sites are hard?)
  results/figures/pred_vs_true_*.png  one day of predictions vs ground truth
  results/figures/loss_curves.png     training/validation loss per model
  results/figures/error_by_hour.png   error by time of day (peak vs off-peak)
  + training time & inference time per model (good "insight" material)

Ideas for a "comprehensive" evaluation (Evaluation = 15 marks):
  - same test set, same lags, same seed for every model
  - multiple runs / seeds -> mean +- std
  - peak hours vs night, weekday vs weekend
  - effect of `lags` (4, 8, 12, 24) - also great for the Research section
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config


def evaluate_all(cfg: dict | None = None) -> pd.DataFrame:
    """Load every saved model, predict on ds.X_test, compute metrics, write CSVs + figures.

    TODO(Sakib)
    """
    cfg = cfg or load_config()
    raise NotImplementedError("Sakib: implement evaluate_all()")
