"""Ronda 3: red multihorizonte (purga por horizonte, máscaras y puntuación de la mediana)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from oilcast.features.windows import Panel
from oilcast.models.recurrent import HParams
from oilcast.training import multitask as mt
from oilcast.training.trainer import fit, prepare_data
from oilcast.training.tune import median_score as single_median_score

H = [5, 63, 252]


def test_train_set_only_uses_targets_realised_before_cutoff(panel) -> None:
    cutoff = pd.Timestamp("2014-01-01")
    rows, known = mt.train_set(panel, H, cutoff)
    _, td = mt.targets(panel, H)
    assert known.any(axis=1).all()
    assert (td[rows][known] < np.datetime64(cutoff)).all()
    # each head keeps exactly the origins the single-horizon purge would keep
    from oilcast.training.splits import purged_train

    for j, h in enumerate(H):
        np.testing.assert_array_equal(rows[known[:, j]], purged_train(panel, h, cutoff))


def test_prepare_masks_unknown_targets(panel, cfg) -> None:
    c = copy.deepcopy(cfg)
    c["targets"]["remove_drift"] = True
    rows, known = mt.train_set(panel, H, "2014-01-01")
    data = mt.prepare(panel, H, rows, known, c, torch.device("cpu"))
    y = data.y.numpy()
    assert np.isfinite(y[rows]).sum() == known.sum()
    outside = np.setdiff1d(np.arange(len(panel)), rows)
    assert not np.isfinite(y[outside]).any()
    for j in range(len(H)):  # centred and scaled with the known training targets only
        assert abs(np.nanmean(y[rows, j])) < 1e-4


def test_multi_pinball_ignores_masked_targets() -> None:
    loss = mt.multi_pinball(2, 3)
    q = torch.tensor([0.1, 0.5, 0.9])
    model = lambda x: torch.zeros(len(x), 6)  # noqa: E731
    target = torch.tensor([[1.0, float("nan")], [1.0, float("nan")]])
    full = torch.tensor([[1.0, 5.0], [1.0, 5.0]])
    only_first = mt.multi_pinball(1, 3)(
        lambda x: torch.zeros(len(x), 3), torch.zeros(2), full[:, :1], q
    )
    assert torch.isclose(loss(model, torch.zeros(2), target, q), only_first)


def test_multitask_fit_and_median_score(panel, cfg) -> None:
    c = copy.deepcopy(cfg)
    c["targets"]["remove_drift"] = True
    folds = mt.cv_folds(panel, H, c)
    f = folds[-1]
    data = mt.prepare(
        panel, H, f["rows"], f["known"], c, torch.device("cpu"), (f["vrows"], f["vknown"])
    )
    hp = HParams(4, 1, 0.0, 1e-3, 1e-6, 512)
    res = fit(
        "gru",
        hp,
        data,
        f["rows"],
        c,
        epochs=2,
        seed=0,
        val_idx=f["vrows"],
        model_factory=mt.factory(3),
        objective=mt.multi_pinball(3, 7),
        validation_score=mt.median_score(3, 7, 3),
    )
    assert np.isfinite(res.best_val) and res.best_val > 0
    # a model that predicts zero return scores exactly 1 (the naive forecast)
    zero = lambda x: torch.zeros(len(x), 21)  # noqa: E731
    zero.eval = lambda: None
    assert abs(mt.median_score(3, 7, 3)(zero, data, f["vrows"]) - 1.0) < 1e-6


def test_single_median_score_naive_is_one(panel, cfg) -> None:
    c = copy.deepcopy(cfg)
    c["targets"]["remove_drift"] = True
    from oilcast.training.splits import purged_train

    train = purged_train(panel, 63, pd.Timestamp("2014-01-01"))
    data = prepare_data(panel, 63, train, c, torch.device("cpu"))
    zero = lambda x: torch.zeros(len(x), 7)  # noqa: E731
    zero.eval = lambda: None
    assert abs(single_median_score(3)(zero, data, train) - 1.0) < 1e-6


def test_product_features(df, cfg) -> None:
    c = copy.deepcopy(cfg)
    c["features"]["products"] = True
    d = df.assign(gasoline=df["brent"] / 42 * 1.2, heating_oil=df["brent"] / 42 * 1.3)
    p = Panel.from_dataset(d, c)
    i = p.feature_names.index("crack_gas")
    np.testing.assert_allclose(p.features[100:, i], np.log(1.2), rtol=1e-5)
