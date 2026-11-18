"""Leakage controls of the temporal splits (purging, ordering, no overlap)."""

import numpy as np
import pandas as pd
import pytest

from oilcast.training.splits import cv_folds, refit_blocks


@pytest.mark.parametrize("h", [5, 63, 252])
def test_cv_folds_are_purged_and_ordered(panel, cfg, h):
    folds = cv_folds(panel, h, cfg)
    assert len(folds) == len(cfg["splits"]["cv_blocks"])
    td = panel.target_dates(h)
    test_start = np.datetime64(pd.Timestamp(cfg["splits"]["test_start"]))
    for fold, (start, end) in zip(folds, cfg["splits"]["cv_blocks"], strict=True):
        start, end = np.datetime64(pd.Timestamp(start)), np.datetime64(pd.Timestamp(end))
        # every training target is realised before the validation block starts
        assert (td[fold.train] < start).all()
        # validation origins lie inside the block, after every training origin
        val_dates = panel.dates.values[fold.val]
        assert (val_dates >= start).all() and (val_dates < end).all()
        assert fold.train.max() < fold.val.min()
        assert not set(fold.train) & set(fold.val)
        # development never sees outcomes from the test period
        assert (td[fold.val] < test_start).all()
        # the purge drops exactly the usable pre-block origins whose target crosses the start
        before = np.flatnonzero(panel.usable(h) & (panel.dates.values < start))
        purged = np.setdiff1d(before, fold.train)
        assert len(purged) > 0 and (td[purged] >= start).all()


@pytest.mark.parametrize("h", [5, 63, 252])
def test_refit_blocks_never_train_on_unrealised_targets(panel, cfg, h):
    blocks = refit_blocks(panel, h, cfg)
    td = panel.target_dates(h)
    assert any(b.phase == "test" for b in blocks)
    seen = set()
    for b in blocks:
        assert (td[b.train] < np.datetime64(b.refit_date)).all()
        dates = panel.dates.values[b.predict]
        assert (dates >= np.datetime64(b.refit_date)).all()
        assert (dates < np.datetime64(b.end_date)).all()
        assert not seen & set(b.predict), "an origin is predicted twice"
        seen |= set(b.predict)
    test_start = np.datetime64(pd.Timestamp(cfg["splits"]["test_start"]))
    calib = [b for b in blocks if b.phase == "calibration"]
    assert all((panel.dates.values[b.predict] < test_start).all() for b in calib)
