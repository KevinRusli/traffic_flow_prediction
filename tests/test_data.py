"""Data processing tests - OWNER: Kevin.  (TC01-TC06, see docs/TEST_CASES.md)"""
import numpy as np
import pandas as pd

from src.data.loader import LONG_COLUMNS, VOLUME_COLUMNS, fix_coordinates, load_raw, to_long
from src.data.preprocess import build_dataset, chronological_split, make_windows


def test_tc01_load_raw_structure(scats_file):
    df = load_raw(scats_file)
    assert set(VOLUME_COLUMNS).issubset(df.columns)
    assert df["scats"].nunique() == 40
    assert not df.duplicated().any(), "duplicate rows (site 4335) must be removed"


def test_tc02_to_long_shape(scats_file):
    wide = load_raw(scats_file)
    long = to_long(wide)
    assert set(LONG_COLUMNS) <= set(long.columns)
    assert len(long) == len(wide) * 96
    assert (long["flow"] >= 0).all()
    one = long[long["location"] == long["location"].iloc[0]].sort_values("timestamp")
    assert (one["timestamp"].diff().dropna().mode()[0]) == pd.Timedelta(minutes=15)


def test_tc03_no_zero_coordinates(scats_file):
    long = fix_coordinates(to_long(load_raw(scats_file)))
    assert (long["lat"] != 0).all() and (long["lon"] != 0).all()


def test_tc04_make_windows_toy():
    X, y = make_windows(np.array([1, 2, 3, 4, 5]), lags=3)
    np.testing.assert_array_equal(X, [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(y, [4, 5])


def test_tc05_split_is_chronological(toy_long_df):
    train, test = chronological_split(toy_long_df, "2006-10-02")
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) == len(toy_long_df)


def test_tc06_build_dataset_shapes(toy_long_df):
    cfg = {"data": {"lags": 12, "horizon": 1, "train_end_date": "2006-10-02", "scaler": "minmax"}}
    ds = build_dataset(toy_long_df, cfg)
    assert ds.X_train.shape[1] == 12 and ds.X_train.shape[0] == ds.y_train.shape[0]
    assert ds.X_test.shape[1] == 12
    assert ds.X_train.min() >= 0 and ds.X_train.max() <= 1, "scaler must be fit on train"
