"""Map model names (as used in config.yaml / GUI / CLI) to model classes.

    from src.models.registry import get_model, get_model_class
    model = get_model("gru")                 # new, untrained, with config params
    model = get_model_class("gru").load(p)   # trained, from disk
"""
from __future__ import annotations

from src.config import load_config
from src.models.base import BaseTrafficModel
from src.models.gru import GRUModel
from src.models.lstm import LSTMModel
from src.models.third_model import ThirdModel

MODEL_CLASSES: dict[str, type[BaseTrafficModel]] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "third": ThirdModel,
}


def get_model_class(name: str) -> type[BaseTrafficModel]:
    try:
        return MODEL_CLASSES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_CLASSES)}") from exc


def get_model(name: str, cfg: dict | None = None) -> BaseTrafficModel:
    cfg = cfg or load_config()
    model_cfg = cfg["models"].get(name, {})
    params = model_cfg.get("params", model_cfg)
    return get_model_class(name)(params=params, training=cfg["models"]["training"])
