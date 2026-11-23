"""Train one or all models and save them to models/saved/.

OWNER: Hassan (shared training loop, used by Ali for GRU and Sakib for Model 3)

    python main.py train --model lstm
    python main.py train --model all

For every model this writes:
    models/saved/<name>.*                  the trained model (format chosen by the model)
    models/saved/scaler.joblib             the scaler, shared by all models
    results/history_<name>.json            loss curves, for the loss-curve figure
    results/training_times.csv             seconds to train each model (for the Insights)

All models use the same dataset, split, lags, seed and shuffled sample order,
so the comparison is fair.
"""
from __future__ import annotations

import json
import random
import time

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.models.base import BaseTrafficModel


def train_model(name: str, cfg: dict | None = None, ds=None) -> BaseTrafficModel:
    """Load data -> build dataset -> fit model -> save model + scaler + history.

    `ds` (a TrafficDataset) can be passed in so train_all() builds it only once.
    """
    from src.data.loader import load_long
    from src.data.preprocess import build_dataset
    from src.models.registry import get_model

    cfg = cfg or load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # fix the python/numpy seeds (Keras models also seed TensorFlow inside fit)
    seed = cfg["models"]["training"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)

    if ds is None:
        ds = build_dataset(load_long(), cfg)
    print(f"[{name}] training on {len(ds.y_train):,} samples, lags = {ds.X_train.shape[1]}")

    # The samples are sorted by location, and Keras takes the LAST part of the data
    # as the validation set. Without shuffling, validation would be only the last
    # ~15 locations (alphabetically). Shuffling with the fixed seed makes it a random
    # sample of all locations, the same for every model.
    order = np.random.default_rng(seed).permutation(len(ds.y_train))
    X_train, y_train = ds.X_train[order], ds.y_train[order]

    model = get_model(name, cfg)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    epochs = len(model.history.get("loss", []))
    print(f"[{name}] done in {seconds:.1f} s" + (f" ({epochs} epochs)" if epochs else ""))

    model.save(models_dir / name)
    # the scaler is fitted on the training data only, so it is the same for every model
    joblib.dump(ds.scaler, models_dir / "scaler.joblib")

    with open(results_dir / f"history_{name}.json", "w", encoding="utf-8") as f:
        json.dump(model.history, f, indent=2)

    # one row per model; re-training a model replaces its old row
    times_file = results_dir / "training_times.csv"
    times = pd.read_csv(times_file) if times_file.exists() else pd.DataFrame(columns=["model", "seconds", "epochs"])
    times = times[times["model"] != name]
    times.loc[len(times)] = [name, round(seconds, 1), epochs]
    times.to_csv(times_file, index=False)

    return model


def train_all(cfg: dict | None = None) -> dict[str, BaseTrafficModel]:
    from src.data.loader import load_long
    from src.data.preprocess import build_dataset

    cfg = cfg or load_config()
    ds = build_dataset(load_long(), cfg)        # build once, reuse for every model
    return {name: train_model(name, cfg, ds) for name in cfg["models"]["available"]}
