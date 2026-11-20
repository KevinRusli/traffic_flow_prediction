"""LSTM model for 15-minute traffic-flow prediction.

OWNER: Hassan (Model 1)

Architecture (same shape as the GRU so the comparison is fair):

    Input (lags, 1) -> LSTM(64, return_sequences) -> LSTM(64) -> Dropout(0.2) -> Dense(1)

  - Input arrives as (n, lags); it is reshaped to (n, lags, 1) inside fit/predict
    because Keras recurrent layers want (samples, time steps, features).
  - Hyper-parameters come from cfg["models"]["lstm"] and cfg["models"]["training"].
  - Early stopping on a validation split; history["loss"] / ["val_loss"] are kept
    for the loss-curve figure in the report.
  - TensorFlow/Keras is imported INSIDE the methods so the rest of the app (and the
    tests) still import when TF isn't installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.base import BaseTrafficModel


class LSTMModel(BaseTrafficModel):
    name = "lstm"

    def build(self, lags: int):
        """Return a compiled Keras model: stacked LSTM layers -> Dropout -> Dense(1)."""
        from tensorflow import keras
        from tensorflow.keras import layers

        units = self.params.get("units", [64, 64])
        dropout = self.params.get("dropout", 0.2)

        model = keras.Sequential()
        model.add(layers.Input(shape=(lags, 1)))
        for i, n in enumerate(units):
            # every LSTM except the last passes its whole sequence to the next LSTM;
            # the last one returns only its final state (one vector per sample)
            model.add(layers.LSTM(n, return_sequences=i < len(units) - 1))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1))      # next 15-minute flow (scaled)

        lr = self.training.get("learning_rate", 0.001)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LSTMModel":
        from tensorflow import keras

        # same seed -> same weights and same validation split every run
        seed = self.training.get("random_seed")
        if seed is not None:
            keras.utils.set_random_seed(seed)

        lags = X_train.shape[1]
        self._model = self.build(lags)
        X3d = X_train.reshape(-1, lags, 1).astype("float32")

        # stop when validation loss stops improving and keep the best epoch's weights
        callbacks = []
        patience = self.training.get("early_stopping_patience")
        if patience:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True))

        history = self._model.fit(
            X3d, y_train.astype("float32"),
            epochs=self.training.get("epochs", 50),
            batch_size=self.training.get("batch_size", 256),
            validation_split=self.training.get("validation_split", 0.1),
            callbacks=callbacks,
            verbose=0,
        )
        # plain floats so train.py can write it to JSON
        self.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X3d = X.reshape(-1, X.shape[1], 1).astype("float32")
        return self._model.predict(X3d, verbose=0).reshape(-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(f"{path}.keras")

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        from tensorflow import keras

        model = cls()
        model._model = keras.models.load_model(f"{Path(path)}.keras")
        return model