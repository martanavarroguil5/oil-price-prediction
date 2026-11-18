"""Causality, aligned comparisons and development-only selection for the simple models."""

from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from oilcast.evaluation.baselines import (
    Candidate,
    candidates,
    choose_by_horizon,
    development_mask,
    read_network,
    ridge_coefficients,
    simple_predictions,
    skill_interval,
)
from oilcast.paths import PROJECT_ROOT
from oilcast.training.splits import purged_train


def baseline_config(cfg):
    out = copy.deepcopy(cfg)
    out["baselines"] = {
        "mean_reversion_windows": [63, 252],
        "ridge_alphas": [0.1, 10.0],
        "simplicity_tolerance": 0.01,
    }
    return out


def test_ridge_learns_a_signal_and_regularisation_shrinks_it():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(500, 4))
    beta = np.array([0.2, -0.4, 0.0, 0.1])
    y = x @ beta
    fitted = ridge_coefficients(x[:400], y[:400], 0.001)
    shrunk = ridge_coefficients(x[:400], y[:400], 100)
    assert np.mean((x[400:] @ fitted - y[400:]) ** 2) < np.mean(y[400:] ** 2) / 100
    assert np.linalg.norm(shrunk) < np.linalg.norm(fitted)
    with pytest.raises(ValueError, match="positive"):
        ridge_coefficients(x, y, 0)


@pytest.mark.parametrize("h", [5, 63, 252])
def test_future_outcomes_and_features_do_not_affect_earlier_fits(panel, cfg, h):
    c = baseline_config(cfg)
    train = purged_train(panel, h, pd.Timestamp("2014-01-01"))
    origins = np.flatnonzero(panel.usable(h) & (panel.dates >= "2014-01-01"))[:10]
    specs = candidates(c)
    original = simple_predictions(panel, h, train, origins, specs, c)
    changed = copy.deepcopy(panel)
    # No training target reaches this date. Only as-yet-unknown outcomes change.
    changed.price[panel.dates >= "2014-01-01"] *= 100
    # Features beyond all the requested origins cannot change their input or scaler.
    changed.features[origins[-1] + 1 :] = 1e8
    after = simple_predictions(changed, h, train, origins, specs, c)
    for name in original:
        np.testing.assert_array_equal(after[name], original[name])


def test_mean_reversion_cannot_turn_into_momentum_or_overshoot(panel, cfg):
    c = baseline_config(cfg)
    changed = copy.deepcopy(panel)
    col = changed.feature_names.index("dev_ma63")
    # A synthetic constant log-return and a negative deviation imply beta > 1.
    changed.price = np.exp(np.arange(len(panel)) * 0.001)
    changed.features[:, col] = -0.001
    train = purged_train(changed, 5, pd.Timestamp("2014-01-01"))
    origins = np.flatnonzero(panel.valid & (panel.dates >= "2014-01-01"))[:4]
    spec = [Candidate("reversion_63", "reversion", 63)]
    positive = simple_predictions(changed, 5, train, origins, spec, c)["reversion_63"]
    np.testing.assert_allclose(positive, 0.001)
    changed.features[:, col] = 0.001  # fitted beta is negative: fall back to zero
    negative = simple_predictions(changed, 5, train, origins, spec, c)["reversion_63"]
    np.testing.assert_array_equal(negative, np.zeros(len(origins)))


def test_development_selection_excludes_unrealised_targets_and_future_results(cfg):
    c = baseline_config(cfg)
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2020-01-01", "2021-12-01", "2022-01-03"]),
            "target_date": pd.to_datetime(["2020-04-01", "2022-03-01", "2022-04-01"]),
            "skill_mae": [0.02, 1.0, 1.0],
        }
    )
    np.testing.assert_array_equal(development_mask(frame, "2022-01-01"), [True, False, False])
    scores = pd.DataFrame(
        [
            {"horizon": 5, "model": "naive", "skill_mae": 0.0},
            {"horizon": 5, "model": "rnn", "skill_mae": 0.009},
            {"horizon": 63, "model": "naive", "skill_mae": 0.0},
            {"horizon": 63, "model": "ridge_10", "skill_mae": 0.03},
            {"horizon": 63, "model": "rnn", "skill_mae": 0.035},
        ]
    )
    decisions = choose_by_horizon(scores, candidates(c), 0.01)
    assert decisions["5"]["model"] == "naive"
    assert decisions["63"]["model"] == "ridge_10"


def test_network_comparison_rejects_missing_dates_and_different_targets(tmp_path):
    origins = pd.bdate_range("2020-01-01", periods=4)
    ref = pd.DataFrame(
        {
            "origin": origins,
            "target_date": origins + pd.Timedelta(days=7),
            "refit_date": pd.Timestamp("2020-01-01"),
            "phase": "calibration",
            "price": 50.0,
            "y": 0.01,
        }
    )
    path = tmp_path / "network.csv"
    saved = ref.assign(q50=0.005)
    saved.to_csv(path, index=False)
    out = read_network(path, ref, "rnn")
    np.testing.assert_array_equal(out["pred_log"], 0.005)
    saved.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Origins differ"):
        read_network(path, ref, "rnn")
    saved.loc[0, "target_date"] += pd.Timedelta(days=1)
    saved.to_csv(path, index=False)
    with pytest.raises(ValueError, match="target_date differs"):
        read_network(path, ref, "rnn")


def test_paired_block_bootstrap_preserves_known_relative_gain():
    # Every paired error is 20% smaller, for any resampled sequence of blocks.
    base = np.linspace(1, 20, 300)
    lo, hi = skill_interval(base * 0.8, base, 63, 100, 42)
    np.testing.assert_allclose([lo, hi], [0.2, 0.2])
    first = skill_interval(base[::-1], base, 63, 100, 42)
    assert first == skill_interval(base[::-1], base, 63, 100, 42)


def test_pipeline_freezes_selection_before_test_and_preserves_production(
    tmp_paths, cfg, monkeypatch
):
    from oilcast.evaluation import baselines

    c = baseline_config(cfg)
    c["architectures"] = []
    c["targets"]["horizons"] = [63]
    c["baselines"].update(
        output_dir="reports/experiments/baselines", bootstrap_samples=10, bootstrap_seed=42
    )
    for relative in (
        "docs/experimento-baselines.md",
        "src/oilcast/evaluation/baselines.py",
        "src/oilcast/evaluation/metrics.py",
        "src/oilcast/features/windows.py",
        "src/oilcast/features/engineering.py",
        "src/oilcast/training/splits.py",
    ):
        destination = tmp_paths.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, destination)
    sentinel = tmp_paths.results / "selection.json"
    sentinel.write_text('{"chosen": "rnn"}', encoding="utf-8")
    output = tmp_paths.root / c["baselines"]["output_dir"]
    original_score = baselines.score
    test_calls = []

    def checked_score(frame, h, config, uncertainty):
        if uncertainty:
            frozen = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            assert "63" in frozen["selected"]
            test_calls.append(frozen["selected"])
        return original_score(frame, h, config, uncertainty)

    monkeypatch.setattr(baselines, "score", checked_score)
    result = baselines.run_baselines(c, tmp_paths)
    assert test_calls and all(selection == result["selected"] for selection in test_calls)
    assert json.loads(sentinel.read_text(encoding="utf-8")) == {"chosen": "rnn"}
    assert (output / "resultado.md").exists()
    audit = pd.read_csv(
        output / "training_audit.csv", parse_dates=["latest_train_target", "refit_date"]
    )
    assert (audit["latest_train_target"] < audit["refit_date"]).all()
