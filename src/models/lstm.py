"""LSTM model for 15-minute traffic-flow prediction.

OWNER: Hassan (Model 1)

Reference implementation to study/adapt:
    https://github.com/hyswin/Example-LSTM-and-GRU-for-A2B-COS30019-

Notes:
  - Input arrives as (n, lags); reshape to (n, lags, 1) inside fit/predict.
  - Hyper-parameters come from cfg["models"]["lstm"] and cfg["models"]["training"].
  - Use early stopping on a validation split; keep history["loss"] / ["val_loss"]
    for the loss-curve figure in the report.
  - Import TensorFlow/Keras INSIDE the methods so the rest of the app (and the
    tests) still import when TF isn't installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.base import BaseTrafficModel


class LSTMModel(BaseTrafficModel):
    name = "lstm"

    def build(self, lags: int):
        """TODO(Hassan): return a compiled Keras model (e.g. LSTM-LSTM-Dropout-Dense)."""
        raise NotImplementedError("Hassan: implement LSTMModel.build()")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LSTMModel":
        raise NotImplementedError("Hassan: implement LSTMModel.fit()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Hassan: implement LSTMModel.predict()")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Hassan: implement LSTMModel.save()  (e.g. path + '.keras')")

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        raise NotImplementedError("Hassan: implement LSTMModel.load()")
