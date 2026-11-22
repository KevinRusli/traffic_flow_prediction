"""Random Forest model for traffic-flow prediction.

OWNER: Sakib Ahmed 

Uses the shared 2-D lag windows and returns scaled next-interval predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseTrafficModel


class ThirdModel(BaseTrafficModel):
    name = "third"

    def __init__(self, params: dict[str, Any] | None = None, training: dict[str, Any] | None = None) -> None:
        super().__init__(params=params, training=training)
        self.model: RandomForestRegressor | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ThirdModel":
        """Train the trees using the shared scaled inputs and targets."""
        # Starting settings; later experiments can tune them on validation data.
        settings = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_leaf": 2,
            "random_state": self.training.get("random_seed", 42),
            "n_jobs": -1,
        }
        settings.update(self.params)
        self.model = RandomForestRegressor(**settings)
        self.model.fit(X_train, y_train)
        # Forests have no epoch-based loss curve, so history stays empty.
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return one scaled traffic-flow prediction per input window."""
        if self.model is None:
            raise ValueError("Train or load the model before predicting.")
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        """Save the fitted forest and its settings to a joblib file."""
        if self.model is None:
            raise ValueError("Train the model before saving.")
        # The shared interface supplies a path without an extension.
        output = Path(path).with_suffix(".joblib")
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "params": self.params,
                "training": self.training,
                "history": self.history,
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ThirdModel":
        """Restore the forest and settings previously written by save()."""
        saved = joblib.load(Path(path).with_suffix(".joblib"))
        instance = cls(params=saved["params"], training=saved["training"])
        instance.model = saved["model"]
        instance.history = saved["history"]
        return instance
