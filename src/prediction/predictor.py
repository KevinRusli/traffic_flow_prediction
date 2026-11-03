"""The bridge between the ML part and the routing part.

OWNER: Kevin (Integration A2A), using the data helpers and the trained models.

The routing code ONLY knows this interface:

    predictor.predict_flow(scats=2000, when=datetime(2006, 10, 15, 8, 0)) -> vehicles per HOUR

That means Kevin can build and test the full routing pipeline with
ConstantPredictor today, and swap in ModelPredictor once the models are trained.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class FlowPredictor(ABC):
    @abstractmethod
    def predict_flow(self, scats: int, when: datetime) -> float:
        """Predicted traffic volume at SCATS site `scats` around `when`, in vehicles/hour."""


class ConstantPredictor(FlowPredictor):
    """Returns the same flow everywhere. Use for tests and early integration work."""

    def __init__(self, flow_vph: float = 200.0) -> None:
        self.flow_vph = flow_vph

    def predict_flow(self, scats: int, when: datetime) -> float:
        return self.flow_vph


class ModelPredictor(FlowPredictor):
    """Uses a trained model (lstm / gru / third) to predict the flow.

    TODO(Kevin):
      __init__: load the model via get_model_class(name).load(...), the scaler,
                and the long DataFrame (src.data.loader.load_long).
      predict_flow:
        1. for every approach `location` of site `scats`:
             window = latest_window(df_long, location, when, lags)   # raw counts
             x = scaler.transform(window) -> model.predict -> scaler.inverse
        2. aggregate the approaches with cfg["prediction"]["site_flow_aggregation"]
        3. convert vehicles/15 min -> vehicles/hour (x 4)
      Cache results per (scats, when) - route search calls this many times.
    """

    def __init__(self, model_name: str, cfg: dict | None = None) -> None:
        self.model_name = model_name
        raise NotImplementedError("Kevin: implement ModelPredictor.__init__()")

    def predict_flow(self, scats: int, when: datetime) -> float:
        raise NotImplementedError("Kevin: implement ModelPredictor.predict_flow()")
