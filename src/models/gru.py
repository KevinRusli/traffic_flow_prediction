"""GRU model for traffic flow prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.base import BaseTrafficModel


class GRUModel(BaseTrafficModel):
    name = "gru"

    def build(self, lags: int):
        # import here so TF isn't required just to import this file
        from tensorflow import keras
        from tensorflow.keras import layers

        units = self.params.get("units", [64, 64])
        dropout = self.params.get("dropout", 0.2)

        # return_sequences=True only needed if there's a second GRU layer after
        model = keras.Sequential([
            layers.Input(shape=(lags, 1)),
            layers.GRU(units[0], return_sequences=len(units) > 1),
            *([layers.GRU(units[1]), layers.Dropout(dropout)] if len(units) > 1
              else [layers.Dropout(dropout)]),
            layers.Dense(1),
        ])
        lr = self.training.get("learning_rate", 0.001)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GRUModel":
        from tensorflow import keras

        lags = X_train.shape[1]
        self._model = self.build(lags)
        X3d = X_train.reshape(-1, lags, 1).astype("float32")  # Keras wants (n, lags, 1)
        y = y_train.astype("float32")

        callbacks = []
        patience = self.training.get("early_stopping_patience")
        if patience:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ))

        seed = self.training.get("random_seed")
        if seed is not None:
            keras.utils.set_random_seed(seed)

        history = self._model.fit(
            X3d, y,
            epochs=self.training.get("epochs", 50),
            batch_size=self.training.get("batch_size", 256),
            validation_split=self.training.get("validation_split", 0.1),
            callbacks=callbacks,
            verbose=0,
        )
        self.history = {k: list(map(float, v)) for k, v in history.history.items()}  # for loss-curve plot
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        lags = X.shape[1]
        X3d = X.reshape(-1, lags, 1).astype("float32")
        preds = self._model.predict(X3d, verbose=0)
        return preds.reshape(-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(f"{path}.keras")

    @classmethod
    def load(cls, path: str | Path) -> "GRUModel":
        from tensorflow import keras

        model = cls()
        model._model = keras.models.load_model(f"{Path(path)}.keras")
        return model
