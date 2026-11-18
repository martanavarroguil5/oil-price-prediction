"""The historical replay never exposes future observations until explicit reveal."""

import numpy as np
import pandas as pd
import pytest

from oilcast.paths import get_paths
from oilcast.replay import replay_case, replay_dates


@pytest.fixture
def replay_paths(tmp_path):
    paths = get_paths({}, root=tmp_path)
    paths.ensure()
    dates = pd.bdate_range("2022-01-03", periods=12)
    prices = np.arange(100.0, 112.0)
    pd.DataFrame({"brent": prices}, index=pd.DatetimeIndex(dates, name="date")).to_csv(
        paths.dataset_csv
    )
    frame = pd.DataFrame(
        {
            "origin": dates[:2],
            "target_date": dates[5:7],
            "refit_date": "2022-01-01",
            "phase": "test",
            "price": prices[:2],
            "q50": np.log([1.03, 1.03]),
            "lo80": np.log([0.95, 0.95]),
            "hi80": np.log([1.10, 1.10]),
            "y": np.log(prices[5:7] / prices[:2]),
        }
    )
    folder = paths.results / "predictions"
    folder.mkdir()
    frame.to_csv(folder / "rnn_h5_ensemble.csv", index=False)
    return paths


def test_before_reveal_has_only_past_data_and_the_saved_forecast(replay_paths):
    record = replay_case(replay_paths, "rnn", 5, "2022-01-04")
    assert not record["revealed"] and "outcome" not in record
    assert all(point[0] <= record["origin"] for point in record["history"])
    assert record["targetDate"] == "2022-01-11"
    assert record["median"] == pytest.approx(104.03)
    assert replay_dates(replay_paths, "rnn", 5)["origins"] == ["2022-01-03", "2022-01-04"]


def test_reveal_is_bounded_by_the_target_and_scores_both_models(replay_paths):
    record = replay_case(replay_paths, "rnn", 5, "2022-01-04", reveal=True)
    outcome = record["outcome"]
    assert record["revealed"]
    assert outcome["actual"] == pytest.approx(106)
    assert outcome["modelError"] == pytest.approx(1.97)
    assert outcome["naiveError"] == pytest.approx(5)
    assert outcome["insideInterval"]
    assert outcome["future"][0][0] == record["origin"]
    assert outcome["future"][-1][0] == record["targetDate"]
    assert all(record["origin"] <= point[0] <= record["targetDate"] for point in outcome["future"])


@pytest.mark.parametrize(
    "arch,horizon,origin",
    [
        ("../../models", 5, "2022-01-03"),
        ("rnn", 7, "2022-01-03"),
        ("rnn", 5, "2022-01-08"),
        ("rnn", 5, "not-a-date"),
    ],
)
def test_replay_rejects_invalid_requests(replay_paths, arch, horizon, origin):
    with pytest.raises(ValueError):
        replay_case(replay_paths, arch, horizon, origin)
