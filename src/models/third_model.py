"""The third ML model (team's choice, needs tutor approval).

OWNER: Sakib (Model 3)

Pick ONE and get it approved by the tutor in class, then set
cfg["models"]["third"]["name"] and its params in config.yaml:

  - Random Forest / Gradient Boosting / XGBoost (sklearn-style, fast, easy to explain)
  - Decision/Regression Tree (the tutor's example in reference/ - fine as a baseline,
    but don't submit the tutor's code itself)
  - SAEs (stacked auto-encoders) - used in the original LSTM/GRU traffic paper
  - TCN / 1-D CNN / Transformer - more research-y, good for the Research section

Tip: also report a naive baseline (e.g. "same time last week") in the evaluation
so the reader can see whether ML actually helps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.base import BaseTrafficModel


class ThirdModel(BaseTrafficModel):
    name = "third"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ThirdModel":
        raise NotImplementedError("Sakib: implement ThirdModel.fit()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Sakib: implement ThirdModel.predict()")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Sakib: implement ThirdModel.save()  (e.g. joblib)")

    @classmethod
    def load(cls, path: str | Path) -> "ThirdModel":
        raise NotImplementedError("Sakib: implement ThirdModel.load()")
