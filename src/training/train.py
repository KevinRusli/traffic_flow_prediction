"""Train one or all models and save them to models/saved/.

OWNER: Hassan (shared training loop, used by Ali for GRU and Sakib for Model 3)

    python main.py train --model lstm
    python main.py train --model all
"""
from __future__ import annotations

from src.config import load_config, resolve_path
from src.models.base import BaseTrafficModel


def train_model(name: str, cfg: dict | None = None) -> BaseTrafficModel:
    """Load data -> build dataset -> fit model -> save model + scaler + history.

    TODO(Hassan):
      from src.data.loader import load_long
      from src.data.preprocess import build_dataset
      from src.models.registry import get_model
      1. ds = build_dataset(load_long(), cfg)
      2. model = get_model(name, cfg); model.fit(ds.X_train, ds.y_train)
      3. model.save(models_dir / name)
      4. save ds.scaler (joblib) to models_dir / "scaler.joblib"  (shared by all models)
      5. save model.history as JSON to results/ for the loss-curve plot
      Fix random seeds (cfg["models"]["training"]["random_seed"]) for reproducibility.
    """
    cfg = cfg or load_config()
    _ = resolve_path(cfg["paths"]["models_dir"])
    raise NotImplementedError("Hassan: implement train_model()")


def train_all(cfg: dict | None = None) -> dict[str, BaseTrafficModel]:
    cfg = cfg or load_config()
    return {name: train_model(name, cfg) for name in cfg["models"]["available"]}
