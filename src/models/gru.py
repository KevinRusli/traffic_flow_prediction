"""GRU model for 15-minute traffic-flow prediction.

OWNER: Ali Murtaza (Model 2)

Same contract as lstm.py - ideally share a helper for the training loop so the
only difference between the two files is the recurrent layer type. That makes
the LSTM vs GRU comparison fair (same lags, epochs, units, optimiser).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.base import BaseTrafficModel


class GRUModel(BaseTrafficModel):
    name = "gru"

    def build(self, lags: int):
        """TODO(Ali Murtaza): return a compiled Keras model (e.g. GRU-GRU-Dropout-Dense)."""
        raise NotImplementedError("Ali Murtaza: implement GRUModel.build()")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GRUModel":
        raise NotImplementedError("Ali Murtaza: implement GRUModel.fit()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Ali Murtaza: implement GRUModel.predict()")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Ali Murtaza: implement GRUModel.save()")

    @classmethod
    def load(cls, path: str | Path) -> "GRUModel":
        raise NotImplementedError("Ali Murtaza: implement GRUModel.load()")
