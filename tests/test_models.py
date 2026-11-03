"""Model tests - OWNERS: Hassan (lstm), Ali Murtaza (gru), Sakib (third + metrics).  (TC11-TC13)"""
import numpy as np
import pytest

from src.evaluation.metrics import all_metrics, nrmse, rmse
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
