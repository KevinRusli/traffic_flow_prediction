"""Compare all trained models on the same test set and produce report material.

OWNER: Sakib (evaluation, feeds the Research section)

    python main.py evaluate

Outputs: metrics.csv, metrics_by_site.csv, metrics_by_hour.csv,
evaluation_notes.txt, and figures/ (all inside the configured results directory).
Every available model uses the same test windows and unscaled target values.
Unavailable models have a status and blank scores, never invented predictions.
Training times and loss curves use records written by the shared training runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd

# Save figures without opening windows, including when run from the CLI or tests.
matplotlib.use("Agg")
from matplotlib import dates as mdates, pyplot as plt  # noqa: E402

from src.config import load_config, resolve_path
from src.data.loader import load_long
from src.data.preprocess import build_dataset
from src.evaluation.metrics import all_metrics
from src.models.registry import get_model_class

METRIC_NAMES = ["MAE", "RMSE", "NRMSE", "MAPE", "R2"]
LABELS = {"lstm": "LSTM", "gru": "GRU", "third": "Random Forest"}


def _group_metrics(frame: pd.DataFrame, columns: list[str], model: str) -> list[dict]:
    """Measure each site or hour using the same metric definitions as the total."""
    rows = []
    for keys, group in frame.groupby(columns, sort=True):
        row = dict(zip(columns, keys))
        row.update(model=model, n_test=len(group))
        row.update(all_metrics(group["actual"].to_numpy(), group["predicted"].to_numpy()))
        rows.append(row)
    return rows


def _save_figure(fig, path: Path) -> None:
    """Save a caption-ready figure and release its memory."""
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_figures(table, frames, histories, hourly, figures_dir: Path) -> bool:
    """Plot available results; return whether saved epoch histories were found."""
    # Remove only our generated charts so an unavailable model cannot leave stale results.
    output_names = ["metrics_comparison.png", "error_by_hour.png", "loss_curves.png"]
    output_names += [f"pred_vs_true_{name}.png" for name in table["model"]]
    for filename in output_names:
        (figures_dir / filename).unlink(missing_ok=True)
    if not frames:
        return False

    available = table[table["status"] == "evaluated"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar([LABELS.get(name, name) for name in available["model"]], available[metric])
        ax.set_title(metric)
        ax.set_ylabel("vehicles / 15 min" if metric != "R2" else "score")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Test results: {len(available)} of {len(table)} models available")
    _save_figure(fig, figures_dir / "metrics_comparison.png")

    # Choose the most complete location/day once, so every prediction plot matches.
    reference = next(iter(frames.values()))
    days = reference["timestamp"].dt.normalize()
    location, day = reference.groupby(["location", days]).size().idxmax()
    for name, frame in frames.items():
        selected = frame[(frame["location"] == location) & (frame["timestamp"].dt.normalize() == day)]
        selected = selected.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(selected["timestamp"], selected["actual"], label="Actual", color="black")
        ax.plot(selected["timestamp"], selected["predicted"], label=LABELS.get(name, name), alpha=0.8)
        ax.set_title(f"{location}\n{day:%Y-%m-%d} - next 15-minute flow")
        ax.set_ylabel("Vehicles / 15 min")
        ax.set_xlabel("Time of day")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend()
        ax.grid(alpha=0.25)
        _save_figure(fig, figures_dir / f"pred_vs_true_{name}.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, group in hourly.groupby("model", sort=False):
        ax.plot(group["hour"], group["RMSE"], marker="o", label=LABELS.get(name, name))
    ax.set(title="Test error by hour of day", xlabel="Hour (local dataset time)", ylabel="RMSE (vehicles / 15 min)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, figures_dir / "error_by_hour.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    has_history = False
    for name, history in histories.items():
        for key in ("loss", "val_loss"):
            values = history.get(key, [])
            if len(values):
                ax.plot(range(1, len(values) + 1), values, label=f"{LABELS.get(name, name)} {key}")
                has_history = True
    if has_history:
        ax.set(title="Saved training histories", xlabel="Epoch", ylabel="Loss (model training units)")
        ax.legend()
        _save_figure(fig, figures_dir / "loss_curves.png")
    else:
        plt.close(fig)  # Random Forest has no epoch loss curve.
    return has_history


def evaluate_all(cfg: dict | None = None) -> pd.DataFrame:
    """Evaluate saved models without retraining and return a table including availability."""
    cfg = cfg or load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    dataset = build_dataset(load_long(), cfg)
    if len(dataset.y_test) == 0 or len(dataset.meta_test) != len(dataset.y_test):
        raise ValueError("Evaluation requires test samples with matching metadata.")

    # Reject a different scaler rather than silently evaluating saved weights in different units.
    saved_scaler = joblib.load(models_dir / "scaler.joblib")
    probes = np.array([[0.0], [1.0], [100.0]])
    if type(saved_scaler) is not type(dataset.scaler) or not np.allclose(
        saved_scaler.transform(probes), dataset.scaler.transform(probes), rtol=1e-7, atol=1e-9
    ):
        raise ValueError("Saved scaler differs from the current dataset. Use the original training configuration.")

    times_file = results_dir / "training_times.csv"
    training_times = {}
    if times_file.is_file():
        training_times = pd.read_csv(times_file).set_index("model")["seconds"].to_dict()

    actual = dataset.inverse(dataset.y_test)
    rows, site_rows, hour_rows = [], [], []
    frames, histories = {}, {}
    for name in cfg["models"]["available"]:
        row = dict(model=name, status="unavailable", reason="", n_test=0)
        row.update({metric: np.nan for metric in METRIC_NAMES})
        row.update(training_time_s=np.nan, inference_time_s=np.nan)
        # Only absent or explicitly unfinished loaders are unavailable. Other errors must surface.
        try:
            model = get_model_class(name).load(models_dir / name)
        except (FileNotFoundError, NotImplementedError) as exc:
            row["reason"] = str(exc)
            rows.append(row)
            continue

        started = perf_counter()
        scaled = np.asarray(model.predict(dataset.X_test))
        elapsed = perf_counter() - started
        if scaled.shape != dataset.y_test.shape:
            raise ValueError(f"{name}: predictions must have shape {dataset.y_test.shape}, got {scaled.shape}.")
        predicted = dataset.inverse(scaled)
        row.update(all_metrics(actual, predicted))
        row.update(status="evaluated", n_test=len(actual), inference_time_s=elapsed)
        row["training_time_s"] = training_times.get(name, np.nan)
        rows.append(row)

        frame = dataset.meta_test.reset_index(drop=True).copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["actual"], frame["predicted"] = actual, predicted
        frame["hour"] = frame["timestamp"].dt.hour
        # Keras model files restore weights but do not include the fit history.
        history_file = results_dir / f"history_{name}.json"
        history = model.history
        if not history and history_file.is_file():
            history = json.loads(history_file.read_text(encoding="utf-8"))
        frames[name], histories[name] = frame, history
        site_rows.extend(_group_metrics(frame, ["scats", "location"], name))
        hour_rows.extend(_group_metrics(frame, ["hour"], name))

    table = pd.DataFrame(rows)
    sites = pd.DataFrame(site_rows, columns=["scats", "location", "model", "n_test"] + METRIC_NAMES)
    hourly = pd.DataFrame(hour_rows, columns=["hour", "model", "n_test"] + METRIC_NAMES)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "metrics.csv", index=False)
    sites.to_csv(results_dir / "metrics_by_site.csv", index=False)
    hourly.to_csv(results_dir / "metrics_by_hour.csv", index=False)
    has_history = _write_figures(table, frames, histories, hourly, figures_dir)
    notes = [
        f"Evaluated {len(frames)} of {len(table)} configured models on the same {len(actual)} test targets.",
        f"Training cutoff: {cfg['data']['train_end_date']}; lags: {cfg['data']['lags']}.",
        "Saved models must have been trained using this configuration and shared scaler.",
        "MAE/RMSE: vehicles per 15 min; NRMSE: RMSE / mean actual flow (ratio).",
        "MAPE: percentage, denominator clipped to at least 1 vehicle, including zero counts.",
        "Inference time covers one predict call on the complete test set; loading is excluded.",
        "Training time is the recorded fit time, when available; it excludes data preparation and saving.",
        "Loss curves use saved histories only." if has_history else "No saved epoch histories; no loss-curve figure.",
    ]
    notes.extend(f"Unavailable {row['model']}: {row['reason']}" for row in rows if row["status"] != "evaluated")
    (results_dir / "evaluation_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return table
