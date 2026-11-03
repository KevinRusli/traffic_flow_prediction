"""Common interface every traffic-flow model must follow.

This file is SHARED. Training, evaluation and the predictor only talk to models
through this interface, so LSTM, GRU and the third model are interchangeable.

    model = get_model("lstm", cfg)          # see registry.py
    model.fit(X_train, y_train)             # X: (n, lags) scaled, y: (n,) scaled
    y_hat = model.predict(X_test)           # -> (n,) scaled
    model.save("models/saved/lstm")         # path WITHOUT extension
    model = LSTMModel.load("models/saved/lstm")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseTrafficModel(ABC):
    #: short id used in config.yaml, file names and result tables
    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None, training: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.training = training or {}
        self.history: dict[str, list[float]] = {}   # loss curves, filled by fit()

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseTrafficModel":
        """Train on 2-D X (n, lags). Store loss curves in self.history if available."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a 1-D array of scaled predictions, same length as X."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained model. `path` has no extension; add your own."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseTrafficModel":
        """Load a model previously written by save()."""
