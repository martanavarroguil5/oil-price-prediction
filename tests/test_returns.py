"""Return metrics use percentage points and a zero-return baseline."""

import numpy as np
import pandas as pd
import pytest

from oilcast.evaluation.returns import return_comparison, return_scores
from oilcast.paths import get_paths


def test_return_errors_are_percentage_points_not_percent_of_actual_return():
    actual = np.array([0.10, -0.05, 0.0])
    predicted = np.array([0.08, -0.04, 0.01])
    metrics = return_scores(actual, predicted)
    assert metrics["mae_pp"] == pytest.approx(4 / 3)
    assert metrics["rmse_pp"] == pytest.approx(np.sqrt(0.0006 / 3) * 100)
    assert metrics["skill_mae"] == pytest.approx(1 - 0.04 / 0.15)
    assert return_scores(np.zeros(3), np.zeros(3))["skill_mae"] is None


def test_saved_log_returns_are_converted_and_dates_must_align(tmp_path):
    paths = get_paths({}, root=tmp_path)
    folder = paths.results / "predictions"
    folder.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "origin": ["2022-01-03", "2022-01-04"],
            "target_date": ["2022-01-10", "2022-01-11"],
            "phase": "test",
            "y": np.log1p([0.1, -0.1]),
            "q50": np.log1p([0.05, -0.05]),
        }
    )
    frame.to_csv(folder / "rnn_h5_ensemble.csv", index=False)
    rows = return_comparison(paths)
    assert rows[0]["model"] == "naive" and rows[0]["mae_pp"] == pytest.approx(10)
    assert rows[1]["model"] == "rnn" and rows[1]["mae_pp"] == pytest.approx(5)
    assert rows[1]["skill_mae"] == pytest.approx(0.5)
    frame.loc[0, "origin"] = "2022-01-02"
    frame.to_csv(folder / "gru_h5_ensemble.csv", index=False)
    with pytest.raises(ValueError, match="identical"):
        return_comparison(paths)
