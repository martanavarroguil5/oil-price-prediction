"""Causal calibration, interval nesting, shared live code and development-only selection."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from oilcast.config import load_config
from oilcast.evaluation.interval_calibration import calibrate_intervals
from oilcast.evaluation.interval_study import choose_interval, development_mask
from oilcast.paths import PROJECT_ROOT, get_paths


@pytest.fixture
def interval_history():
    rng = np.random.default_rng(4)
    origins = pd.bdate_range("2015-01-01", periods=500)
    median = rng.normal(0, 0.02, len(origins))
    frame = pd.DataFrame(
        {
            "origin": origins,
            "target_date": origins + pd.Timedelta(days=30),
            "q50": median,
            "y": median + rng.normal(0, 0.04, len(origins)),
        }
    )
    for level, lo, hi, width in [(50, "q25", "q75", 0.02), (80, "q10", "q90", 0.04)]:
        frame[lo], frame[hi] = median - width, median + width
        frame[f"lo{level}"], frame[f"hi{level}"] = median - width - 0.01, median + width + 0.01
    return frame


@pytest.mark.parametrize("kind", ["cqr", "scaled", "residual"])
def test_calibration_ignores_unrealised_outcomes_and_nests_intervals(interval_history, kind):
    frame = interval_history
    candidate = {"kind": kind, "window": 150}
    before = calibrate_intervals(frame, frame, candidate, min_scores=30)
    changed = frame.copy()
    changed.loc[400:, "y"] = 100
    after = calibrate_intervals(changed, frame, candidate, min_scores=30)
    available = frame["origin"] <= frame.loc[400, "target_date"]
    np.testing.assert_array_equal(before.loc[available], after.loc[available])
    assert (before["lo80"] <= before["lo50"]).all()
    assert (before["lo50"] <= frame["q50"]).all()
    assert (before["hi50"] >= frame["q50"]).all()
    assert (before["hi80"] >= before["hi50"]).all()
    assert before["calibration_n"].max() == 150


def test_live_queries_match_the_historical_calculation_without_targets(interval_history):
    candidate = {"kind": "scaled", "window": None}
    historical = calibrate_intervals(interval_history, interval_history, candidate, 30)
    query = interval_history.iloc[[-1]].drop(columns=["y", "target_date"])
    live = calibrate_intervals(interval_history, query, candidate, 30)
    np.testing.assert_array_equal(live.iloc[0], historical.iloc[-1])
    raw = calibrate_intervals(interval_history, query, {"kind": "current"}, 30)
    np.testing.assert_array_equal(raw.iloc[0], query[["lo50", "hi50", "lo80", "hi80"]].iloc[0])


def test_normalised_cqr_is_equivariant_to_log_return_scale(interval_history):
    spec = {"kind": "scaled", "window": None}
    first = calibrate_intervals(interval_history, interval_history, spec, 30)
    changed = interval_history.copy()
    numeric = changed.select_dtypes(include="number").columns
    changed[numeric] *= 2
    second = calibrate_intervals(changed, changed, spec, 30)
    for col in ("lo50", "hi50", "lo80", "hi80"):
        np.testing.assert_allclose(second[col], first[col] * 2)


def test_selection_rejects_a_narrow_but_undercovering_interval():
    settings = load_config("configs/intervals.yaml")["interval_study"]
    development = pd.DataFrame(
        [
            {
                "candidate": "current",
                "coverage80": 0.8,
                "coverage50": 0.5,
                "winkler80": 100,
                "winkler50": 60,
            },
            {
                "candidate": "cqr_504",
                "coverage80": 0.6,
                "coverage50": 0.4,
                "winkler80": 70,
                "winkler50": 40,
            },
            {
                "candidate": "scaled_expanding",
                "coverage80": 0.79,
                "coverage50": 0.49,
                "winkler80": 95,
                "winkler50": 59,
            },
        ]
    )
    assert choose_interval(development, settings) == "scaled_expanding"
    assert choose_interval(development.iloc[:2], settings) == "current"


def test_development_excludes_targets_realised_after_the_cutoff():
    cfg = load_config("configs/intervals.yaml")
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2015-01-01", "2020-01-01", "2021-12-01", "2022-01-01"]),
            "target_date": pd.to_datetime(["2015-04-01", "2020-04-01", "2022-03-01", "2022-04-01"]),
        }
    )
    np.testing.assert_array_equal(development_mask(frame, cfg), [False, True, False, False])


def test_complete_study_freezes_selection_and_previews_without_modifying_forecasts(
    tmp_path, monkeypatch
):
    from oilcast.evaluation import interval_study
    from oilcast.inference.forecaster import HorizonForecast
    from oilcast.inference.interval_preview import interval_preview

    cfg = load_config("configs/intervals.yaml")
    cfg["interval_study"]["bootstrap_samples"] = 10
    paths = get_paths(cfg, root=tmp_path)
    paths.ensure()
    for relative in (
        "docs/experimento-intervalos.md",
        "src/oilcast/evaluation/interval_calibration.py",
        "src/oilcast/evaluation/interval_study.py",
        "src/oilcast/evaluation/metrics.py",
        "src/oilcast/evaluation/conformal.py",
        "src/oilcast/evaluation/baselines.py",
    ):
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, dst)
    dates = pd.date_range("2014-01-01", "2025-12-01", periods=220)
    rng = np.random.default_rng(10)
    directory = paths.results / "predictions"
    directory.mkdir(parents=True)
    for h in (5, 63, 252):
        frame = pd.DataFrame(
            {
                "origin": dates,
                "target_date": dates + pd.Timedelta(days=h),
                "phase": np.where(dates < "2022-01-01", "calibration", "test"),
                "price": 100.0,
                "y": rng.normal(0, 0.03, len(dates)),
                "q50": 0.0,
                "q25": -0.025,
                "q75": 0.025,
                "q10": -0.05,
                "q90": 0.05,
                "lo50": -0.1,
                "hi50": 0.1,
                "lo80": -0.2,
                "hi80": 0.2,
            }
        )
        for model in ("rnn", "lstm", "gru", "naive"):
            suffix = "" if model == "naive" else "_ensemble"
            frame.to_csv(directory / f"{model}_h{h}{suffix}.csv", index=False)
    sentinel = paths.models / "metadata.json"
    sentinel.write_text('{"untouched": true}', encoding="utf-8")
    output = paths.reports / "experiments/intervals"
    original = interval_study.metrics
    observed = []

    def checked_metrics(frame):
        if (frame["origin"] >= pd.Timestamp("2022-01-01")).all():
            frozen = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            observed.append(frozen["selected"])
        return original(frame)

    monkeypatch.setattr(interval_study, "metrics", checked_metrics)
    result = interval_study.run_interval_study(cfg, paths)
    assert observed and all(choice == result["selected"] for choice in observed)
    assert json.loads(sentinel.read_text(encoding="utf-8")) == {"untouched": True}
    selected = pd.read_csv(output / "predictions/rnn_h252.csv")
    np.testing.assert_array_equal(selected["q50"], np.zeros(len(dates)))
    forecast = HorizonForecast(
        horizon=252,
        origin_date=pd.Timestamp("2026-01-02"),
        target_date=pd.Timestamp("2027-01-02"),
        last_price=100,
        median=110,
        quantiles={0.1: 95, 0.25: 100, 0.5: 110, 0.75: 120, 0.9: 130},
        intervals={"50": (99, 121), "80": (90, 140)},
    )
    preview = interval_preview(paths, {252: forecast}, "rnn")
    assert not preview["stale"]
    lo, hi = preview["latest"]["252"]["experimental"]["80"]
    assert 0 < lo <= forecast.median <= hi
    assert forecast.intervals == {"50": (99, 121), "80": (90, 140)}
    source = directory / "rnn_h252_ensemble.csv"
    changed = pd.read_csv(source)
    changed.loc[0, "q50"] = 0.01
    changed.to_csv(source, index=False)
    assert interval_preview(paths, {252: forecast}, "rnn")["stale"]
