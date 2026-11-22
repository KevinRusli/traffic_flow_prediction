"""Model tests - OWNERS: Hassan (lstm), Ali Murtaza (gru), Sakib (third + metrics).  (TC11-TC13)"""
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.config import load_config
from src.data.preprocess import TrafficDataset
from src.evaluation.metrics import all_metrics, mape, nrmse, r2, rmse
from src.models.registry import get_model, get_model_class

FAST = {"epochs": 2, "batch_size": 32, "learning_rate": 0.01, "validation_split": 0.1,
        "early_stopping_patience": 1, "random_seed": 42}


@pytest.mark.parametrize("name", ["lstm", "gru", "third"])
def test_tc11_fit_predict_shape(name, toy_xy):
    X, y = toy_xy
    model = get_model(name)
    model.training = FAST
    model.fit(X, y)
    pred = model.predict(X[:10])
    assert pred.shape == (10,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", ["lstm", "gru", "third"])
def test_tc12_save_load_roundtrip(name, toy_xy, tmp_path):
    X, y = toy_xy
    model = get_model(name)
    model.training = FAST
    model.fit(X, y)
    model.save(tmp_path / name)
    loaded = get_model_class(name).load(tmp_path / name)
    np.testing.assert_allclose(model.predict(X[:5]), loaded.predict(X[:5]), rtol=1e-5)


def test_tc13_metrics_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 33.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(3.0))
    assert nrmse(y_true, y_pred) == pytest.approx(np.sqrt(3.0) / 20.0)
    m = all_metrics(y_true, y_true)
    assert m["RMSE"] == 0 and m["MAE"] == 0 and m["R2"] == pytest.approx(1.0)


def test_metrics_all_scores_known_values():
    """Sakib: verify every score using three hand-calculated prediction errors."""
    scores = all_metrics(np.array([10.0, 20.0, 30.0]), np.array([10.0, 20.0, 33.0]))
    assert scores == pytest.approx({
        "MAE": 1.0,
        "RMSE": np.sqrt(3.0),
        "NRMSE": np.sqrt(3.0) / 20.0,
        "MAPE": 100.0 / 30.0,
        "R2": 0.955,
    })


def test_metrics_mape_includes_zero_counts_with_denominator_floor():
    """Sakib: a missed zero count must contribute an error, not disappear."""
    actual = np.array([0.0, 10.0])
    predicted = np.array([2.0, 8.0])
    assert mape(actual, predicted) == pytest.approx(110.0)
    assert mape(actual, predicted, eps=2.0) == pytest.approx(60.0)


def test_metrics_nrmse_zero_mean_is_undefined():
    """Sakib: zero traffic cannot provide a normalising mean."""
    assert np.isnan(nrmse(np.zeros(3), np.zeros(3)))
    assert np.isnan(nrmse(np.zeros(3), np.ones(3)))


@pytest.mark.parametrize("predicted, expected", [([5.0, 5.0], 1.0), ([4.0, 6.0], 0.0)])
def test_metrics_r2_constant_actual_values(predicted, expected):
    """Sakib: handle the zero-variance case explicitly."""
    assert r2(np.array([5.0, 5.0]), np.array(predicted)) == expected


def test_metrics_r2_single_sample_is_undefined():
    """Sakib: one observation cannot establish explained variation."""
    assert np.isnan(r2(np.array([5.0]), np.array([5.0])))


@pytest.mark.parametrize("actual, predicted", [
    ([1.0, 2.0], [1.0]),
    ([[1.0], [2.0]], [1.0, 2.0]),
    ([], []),
    ([np.nan], [1.0]),
    ([1.0], [np.inf]),
])
def test_metrics_reject_invalid_pairs(actual, predicted):
    """Sakib: prevent silent broadcasting and invalid evaluation scores."""
    with pytest.raises(ValueError):
        all_metrics(np.array(actual), np.array(predicted))


@pytest.mark.parametrize("eps", [0.0, -1.0, np.nan, np.inf])
def test_metrics_mape_rejects_invalid_floor(eps):
    """Sakib: the denominator floor must prevent division by zero."""
    with pytest.raises(ValueError):
        mape(np.array([1.0]), np.array([1.0]), eps=eps)


@pytest.fixture
def comparison_setup(tmp_path, monkeypatch):
    """Sakib: small saved-model stand-ins with known errors and a real scaler."""
    from src.evaluation import compare

    cfg = load_config()
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    scaler = MinMaxScaler().fit(np.array([[0.0], [100.0]]))
    X = np.ones((4, cfg["data"]["lags"]))
    y = np.array([0.1, 0.2, 0.3, 0.4])
    metadata = pd.DataFrame({
        "scats": [1, 1, 2, 2],
        "location": ["A", "A", "B", "B"],
        "timestamp": pd.date_range("2006-10-25", periods=4, freq="15min"),
    })
    dataset = TrafficDataset(X, y, X, y, scaler, meta_test=metadata)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(scaler, models_dir / "scaler.joblib")
    calls, failures = [], {}

    def model_class(name):
        class Loader:
            @classmethod
            def load(cls, path):
                if name in failures:
                    raise failures[name]

                def predict(inputs):
                    assert inputs is dataset.X_test  # Every model sees the same sample order.
                    calls.append(name)
                    return dataset.y_test + 0.01  # Exactly one vehicle error after inverse scaling.

                history = {"loss": [0.2, 0.1], "val_loss": [0.3, 0.2]} if name == "lstm" else {}
                return SimpleNamespace(predict=predict, history=history)

        return Loader

    monkeypatch.setattr(compare, "load_long", lambda: None)
    monkeypatch.setattr(compare, "build_dataset", lambda data, settings: dataset)
    monkeypatch.setattr(compare, "get_model_class", model_class)
    return compare, cfg, tmp_path / "results", calls, failures


def test_comparison_same_targets_unscaled_scores_and_outputs(comparison_setup):
    """Sakib: compare all three models in original traffic units and produce report files."""
    compare, cfg, results, calls, _ = comparison_setup
    table = compare.evaluate_all(cfg)
    assert calls == ["lstm", "gru", "third"]
    assert table["status"].tolist() == ["evaluated"] * 3
    assert table["n_test"].tolist() == [4] * 3
    np.testing.assert_allclose(table["MAE"], 1.0)
    np.testing.assert_allclose(table["RMSE"], 1.0)
    assert table["training_time_s"].isna().all()
    assert (table["inference_time_s"] >= 0).all()
    saved = pd.read_csv(results / "metrics.csv")
    np.testing.assert_allclose(saved["MAE"], table["MAE"])
    assert len(pd.read_csv(results / "metrics_by_site.csv")) == 6
    assert len(pd.read_csv(results / "metrics_by_hour.csv")) == 3
    for filename in ["metrics_comparison.png", "error_by_hour.png", "loss_curves.png",
                     "pred_vs_true_lstm.png", "pred_vs_true_gru.png", "pred_vs_true_third.png"]:
        assert (results / "figures" / filename).stat().st_size > 0


def test_comparison_reports_missing_models_without_invented_scores(comparison_setup):
    """Sakib: available models still work; stale unavailable-model charts are removed."""
    compare, cfg, results, calls, failures = comparison_setup
    failures.update(lstm=NotImplementedError("LSTM unfinished"), gru=FileNotFoundError("GRU not trained"))
    figures = results / "figures"
    figures.mkdir(parents=True)
    (figures / "pred_vs_true_lstm.png").write_bytes(b"previous run")
    (figures / "loss_curves.png").write_bytes(b"previous run")
    table = compare.evaluate_all(cfg).set_index("model")
    assert calls == ["third"]
    assert table.loc["lstm", "status"] == "unavailable"
    assert table.loc["gru", "reason"] == "GRU not trained"
    assert table.loc[["lstm", "gru"], compare.METRIC_NAMES].isna().all().all()
    assert table.loc["third", "MAE"] == pytest.approx(1.0)
    assert not (figures / "pred_vs_true_lstm.png").exists()
    assert not (figures / "loss_curves.png").exists()
    assert "Evaluated 1 of 3" in (results / "evaluation_notes.txt").read_text()


def test_comparison_does_not_hide_broken_saved_models(comparison_setup):
    """Sakib: corrupt or incompatible models must fail clearly rather than look absent."""
    compare, cfg, _, _, failures = comparison_setup
    failures["lstm"] = ValueError("Corrupt model")
    with pytest.raises(ValueError, match="Corrupt model"):
        compare.evaluate_all(cfg)


def test_comparison_rejects_different_saved_scaler(comparison_setup):
    """Sakib: detect a changed training scale before any model is scored."""
    compare, cfg, results, calls, _ = comparison_setup
    other_scaler = MinMaxScaler().fit(np.array([[0.0], [200.0]]))
    joblib.dump(other_scaler, results.parent / "models" / "scaler.joblib")
    with pytest.raises(ValueError, match="Saved scaler differs"):
        compare.evaluate_all(cfg)
    assert calls == []
