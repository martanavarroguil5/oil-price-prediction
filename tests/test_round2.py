"""Round 2: training without drift and online shrinkage of the central forecast."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from oilcast.evaluation.conformal import online_shrink
from oilcast.evaluation.summary import recalibrate_center
from oilcast.training.splits import purged_train
from oilcast.training.trainer import prepare_data


def test_remove_drift_centres_the_training_target(panel, cfg) -> None:
    h = 63
    train = purged_train(panel, h, pd.Timestamp("2014-01-01"))
    c = copy.deepcopy(cfg)
    c["targets"]["remove_drift"] = True
    data = prepare_data(panel, h, train, c, torch.device("cpu"))
    assert abs(float(data.y[train].mean())) < 1e-4
    plain = prepare_data(panel, h, train, cfg, torch.device("cpu"))
    assert data.y_scale == plain.y_scale


def test_online_shrink_uses_only_known_outcomes() -> None:
    n = 400
    origins = pd.bdate_range("2010-01-01", periods=n).values
    target_dates = origins + np.timedelta64(30, "D")
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 0.05, n)
    y = 0.5 * pred + rng.normal(0, 0.01, n)
    lam = online_shrink(origins, target_dates, pred, y, min_scores=100)
    known = np.searchsorted(np.sort(target_dates), origins, side="left")
    assert np.all(lam[known < 100] == 0.0)  # naive until enough evidence
    assert np.all((lam >= 0) & (lam <= 1))
    assert abs(lam[-1] - 0.5) < 0.05
    # changing an outcome that is not yet known at an origin cannot change its factor
    y2 = y.copy()
    y2[-1] = 10.0
    lam2 = online_shrink(origins, target_dates, pred, y2, min_scores=100)
    np.testing.assert_array_equal(lam[:-1], lam2[:-1])


def test_recalibrate_center_shifts_the_whole_distribution(cfg) -> None:
    c = copy.deepcopy(cfg)
    c["evaluation"].update(recalibrate_center=True, center_min_scores=1)
    origins = pd.bdate_range("2015-01-01", periods=300)
    ens = pd.DataFrame(
        {
            "origin": origins,
            "target_date": origins + pd.Timedelta(days=10),
            "y": np.zeros(300),  # the network never helps -> lambda goes to 0
        }
    )
    qcols = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    base = np.linspace(-0.1, 0.1, 7) + 0.03
    for col, v in zip(qcols, base, strict=True):
        ens[col] = v
    out = recalibrate_center(ens, c)
    late = out.iloc[-1]
    assert late["lam"] == 0.0
    assert abs(late["q50"]) < 1e-12  # median now equals the naive forecast
    widths = out[qcols].to_numpy() - out[["q50"]].to_numpy()
    np.testing.assert_allclose(widths[-1], base - base[3])  # spread unchanged
