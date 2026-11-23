"""Train and check Random Forest using the existing data pipeline.

OWNER: Sakib Ahmed

Run from the project root: python -m src.evaluation.run_random_forest
This runner works while the team's shared training loop is unfinished.
"""

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.loader import load_long
from src.data.preprocess import build_dataset
from src.evaluation.metrics import all_metrics
from src.models.third_model import ThirdModel


def main() -> None:
    """Train Model 3, print test errors, save it, and check reloading."""
    cfg = load_config()

    # Kevin's pipeline creates windows and separates earlier/later target dates.
    print("Loading and preparing traffic data...", flush=True)
    dataset = build_dataset(load_long(), cfg)
    print("Training input shape:", dataset.X_train.shape)
    print("Testing input shape:", dataset.X_test.shape)

    model = ThirdModel(
        params=cfg["models"]["third"]["params"],
        training=cfg["models"]["training"],
    )
    print("Training Random Forest...", flush=True)
    model.fit(dataset.X_train, dataset.y_train)
    predicted_scaled = model.predict(dataset.X_test)

    # Measure errors in vehicles per 15 minutes rather than scaled values.
    actual = dataset.inverse(dataset.y_test)
    predicted = dataset.inverse(predicted_scaled)
    scores = all_metrics(actual, predicted)
    print(f"\nMAE:   {scores['MAE']:.2f} vehicles per 15 minutes")
    print(f"RMSE:  {scores['RMSE']:.2f} vehicles per 15 minutes")
    print(f"NRMSE: {scores['NRMSE']:.4f} (ratio)")
    print(f"MAPE:  {scores['MAPE']:.2f}% (denominator floor: 1 vehicle)")
    print(f"R2:    {scores['R2']:.4f}")

    comparison = pd.DataFrame({"Actual": actual[:10], "Predicted": predicted[:10]})
    print("\nFirst 10 test predictions:")
    print(comparison.to_string(index=False))

    # Store both artifacts where the team's predictor expects to find them.
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    model.save(models_dir / "third")
    joblib.dump(dataset.scaler, models_dir / "scaler.joblib")

    restored = ThirdModel.load(models_dir / "third")
    np.testing.assert_allclose(
        model.predict(dataset.X_test[:10]),
        restored.predict(dataset.X_test[:10]),
    )
    print("\nModel saved:", models_dir / "third.joblib")
    print("Save/load prediction check passed.")


if __name__ == "__main__":
    main()
