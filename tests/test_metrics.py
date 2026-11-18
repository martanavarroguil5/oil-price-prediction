"""Metrics, conformal calibration and the naive baseline."""

import numpy as np
import pandas as pd

from oilcast.evaluation.conformal import conformal_offset, cqr_scores, online_offsets
from oilcast.evaluation.metrics import (
    diebold_mariano,
    directional_accuracy,
    interval_metrics,
    pinball,
    point_metrics,
)
from oilcast.models.naive import naive_quantiles
from oilcast.training.evaluate import naive_predictions


def test_point_metrics():
    m = point_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 1.0]))
    assert np.isclose(m["mae"], 1.0)
    assert np.isclose(m["mse"], 5 / 3)
    assert np.isclose(m["rmse"], np.sqrt(5 / 3))
    assert np.isclose(m["r2"], -1.5)
    assert np.isnan(point_metrics(np.ones(3), np.zeros(3))["r2"])


def test_directional_accuracy_and_naive_is_undefined():
    p0 = np.array([10.0, 10.0, 10.0, 10.0])
    actual = np.array([11.0, 9.0, 12.0, 8.0])
    assert directional_accuracy(p0, actual, np.array([10.5, 9.5, 9.0, 9.0])) == 0.75
    assert np.isnan(directional_accuracy(p0, actual, p0))


def test_pinball_and_intervals():
    y = np.array([0.0, 1.0])
    q = np.array([[0.0, 0.0], [0.0, 2.0]])
    assert np.isclose(pinball(y, q, [0.1, 0.9]), (0 + 0 + 0.1 + 0.1) / 4)
    im = interval_metrics(np.array([1.0, 5.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0]), 0.8)
    assert im["coverage"] == 0.5 and im["width"] == 2.0
    assert np.isclose(im["winkler"], 2 + (2 / 0.2) * 3 / 2)


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 2, 2000)
    good = rng.normal(0, 1, 2000)
    res = diebold_mariano(good, base, h=5)
    assert res["dm_stat"] < 0 and res["dm_pvalue"] < 0.01
    same = diebold_mariano(base, base + 1e-9, h=5)
    assert same["dm_pvalue"] > 0.05 or np.isnan(same["dm_pvalue"])


def test_conformal_uses_only_realised_scores():
    origins = pd.date_range("2020-01-01", periods=300, freq="D").to_numpy()
    targets = origins + np.timedelta64(10, "D")
    scores = np.linspace(-1, 1, 300)
    off = online_offsets(origins, targets, scores, 0.8, window=1000, min_scores=5)
    # the first 15 origins have fewer than 5 realised scores -> no offset
    assert (off[:15] == 0).all()
    # changing scores that are not yet realised at origin i must not change offset i
    scores2 = scores.copy()
    scores2[200:] = 99
    off2 = online_offsets(origins, targets, scores2, 0.8, window=1000, min_scores=5)
    np.testing.assert_allclose(off[:210], off2[:210])
    assert conformal_offset(np.arange(100.0), 0.8) >= 80
    np.testing.assert_allclose(cqr_scores(np.array([0.0]), np.array([1.0]), np.array([2.0])), [1.0])


def test_naive_baseline(panel, cfg):
    q = naive_quantiles(np.array([-0.2, -0.1, 0.0, 0.1, 0.3]), cfg["targets"]["quantiles"])
    assert q[cfg["targets"]["quantiles"].index(0.5)] == 0.0
    assert np.all(np.diff(q) >= 0)
    nv = naive_predictions(cfg, panel, 63)
    assert (nv["q50"] == 0).all()
    assert len(nv) == len(nv["origin"].unique())
