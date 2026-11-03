"""Flow predictors: the link between the ML models and route finding.

OWNER: Kevin

Routing only uses predict_flow(scats, when), which returns vehicles per hour.
ConstantPredictor is for tests; ModelPredictor uses a trained model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.preprocess import latest_window

SCALER_FILE = "scaler.joblib"


class FlowPredictor(ABC):
    @abstractmethod
    def predict_flow(self, scats: int, when: datetime) -> float:
        """Predicted traffic volume at site `scats` around `when`, in vehicles/hour."""


class ConstantPredictor(FlowPredictor):
    """Same flow everywhere, for tests and for routing without a trained model."""

    def __init__(self, flow_vph: float = 200.0) -> None:
        self.flow_vph = flow_vph

    def predict_flow(self, scats: int, when: datetime) -> float:
        return self.flow_vph


class ModelPredictor(FlowPredictor):
    """Predict a site's flow with a trained model (lstm, gru or third).

    Each approach (location) of the site gets its own 15-minute prediction from
    the previous `lags` intervals. The approaches are combined with
    prediction.site_flow_aggregation and scaled to vehicles per hour.

    model, scaler and df_long can be passed in directly (used by the tests);
    otherwise they are loaded from models/saved/ and the dataset.
    """

    def __init__(self, model_name: str, cfg: dict | None = None,
                 model=None, scaler=None, df_long: pd.DataFrame | None = None) -> None:
        self.cfg = cfg or load_config()
        self.model_name = model_name
        models_dir = resolve_path(self.cfg["paths"]["models_dir"])

        if model is None:
            from src.models.registry import get_model_class
            model = get_model_class(model_name).load(models_dir / model_name)
        if scaler is None:
            scaler = joblib.load(models_dir / SCALER_FILE)
        if df_long is None:
            from src.data.loader import load_long
            df_long = load_long()

        self.model = model
        self.scaler = scaler
        self.lags = self.cfg["data"]["lags"]
        self.per_hour = 60 / self.cfg["data"]["interval_minutes"]
        self.aggregate = {"mean": np.mean, "sum": np.sum}[self.cfg["prediction"]["site_flow_aggregation"]]
        self._by_location = {loc: g for loc, g in df_long.groupby("location")}
        self._locations = df_long.groupby("scats")["location"].unique().to_dict()
        self._cache: dict[tuple[int, pd.Timestamp], float] = {}

    def predict_flow(self, scats: int, when: datetime) -> float:
        when = pd.Timestamp(when).floor(f"{self.cfg['data']['interval_minutes']}min")
        key = (scats, when)
        if key not in self._cache:
            self._cache[key] = self._predict(scats, when)
        return self._cache[key]

    def _predict(self, scats: int, when: pd.Timestamp) -> float:
        if scats not in self._locations:
            raise ValueError(f"No traffic data for SCATS site {scats}")
        windows = np.array([
            latest_window(self._by_location[loc], loc, when, self.lags)
            for loc in self._locations[scats]
        ])
        scaled = self.scaler.transform(windows.reshape(-1, 1)).reshape(windows.shape)
        predicted = self.model.predict(scaled.astype("float32"))
        per_interval = self.scaler.inverse_transform(np.reshape(predicted, (-1, 1))).ravel()
        per_interval = np.clip(per_interval, 0, None)
        return float(self.aggregate(per_interval) * self.per_hour)
