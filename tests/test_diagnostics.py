"""Plots preserve genuine OHLC and compare paired errors on the same observations."""

import numpy as np
import pandas as pd
import pytest

from oilcast.evaluation.diagnostics import candle_data, error_samples
from oilcast.paths import get_paths


def test_candles_drop_invalid_ohlc_without_filling_missing_sessions(tmp_path):
    paths = get_paths({}, root=tmp_path)
    paths.ensure()
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
            "open": [90, 91, np.nan, 92],
            "high": [95, 89, 95, 95],
            "low": [88, 88, 88, 91],
            "close": [93, 94, 92, 94],
            "volume": 10,
        }
    ).to_csv(paths.raw / "brent_fut.csv", index=False)
    frame, excluded = candle_data(paths)
    assert excluded == 2
    assert frame.index.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-07"]
    assert frame["open"].tolist() == [90, 92]


@pytest.fixture
def prediction_paths(tmp_path):
    paths = get_paths({}, root=tmp_path)
    directory = paths.results / "predictions"
    directory.mkdir(parents=True)
    for model in ("naive", "rnn", "lstm", "gru"):
        frame = pd.DataFrame(
            {
                "origin": ["2022-01-03", "2022-01-04"],
                "target_date": ["2022-01-10", "2022-01-11"],
                "phase": "test",
                "price": [100, 200],
                "y": np.log1p([0.1, -0.1]),
                "q50": np.zeros(2) if model == "naive" else np.log1p([0.05, -0.05]),
            }
        )
        suffix = "" if model == "naive" else "_ensemble"
        frame.to_csv(directory / f"{model}_h5{suffix}.csv", index=False)
    return paths


def test_violin_errors_use_common_outcomes_and_correct_units(prediction_paths):
    errors = error_samples(prediction_paths, 5)
    rnn = errors[errors["model"] == "rnn"]
    np.testing.assert_allclose(rnn["error_price"], [-5, 10])
    np.testing.assert_allclose(rnn["error_return_pp"], [-5, 5])
    naive = errors[errors["model"] == "naive"]
    np.testing.assert_allclose(naive["error_price"], [-10, 20])
    assert errors.groupby("model").size().eq(2).all()


def test_violin_comparison_rejects_mismatched_samples(prediction_paths):
    path = prediction_paths.results / "predictions/gru_h5_ensemble.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "target_date"] = "2022-01-12"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="identical"):
        error_samples(prediction_paths, 5)
