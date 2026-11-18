"""Purged walk-forward splits.

Purging rule: a sample may be used for training before a cut-off date only if its target is
already realised strictly before that date (``target_date < cutoff``). Because targets span
h trading days, this removes the last h origins before every validation/test block, which
would otherwise share future prices with the block they must not see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oilcast.config import Config
from oilcast.features.windows import Panel


@dataclass
class Fold:
    name: str
    train: np.ndarray
    val: np.ndarray


@dataclass
class RefitBlock:
    refit_date: pd.Timestamp
    end_date: pd.Timestamp
    phase: str  # "calibration" (pre-test, only for conformal) or "test"
    train: np.ndarray
    predict: np.ndarray


def purged_train(panel: Panel, h: int, cutoff: pd.Timestamp) -> np.ndarray:
    """Origins whose target is realised strictly before ``cutoff``."""
    td = panel.target_dates(h)
    ok = panel.usable(h) & (td < np.datetime64(cutoff))
    return np.flatnonzero(ok)


def origins_between(
    panel: Panel,
    h: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    realised_before: pd.Timestamp | None = None,
) -> np.ndarray:
    """Usable origins dated in [start, end), optionally with target realised before a date."""
    d = panel.dates.values
    ok = panel.usable(h) & (d >= np.datetime64(start)) & (d < np.datetime64(end))
    if realised_before is not None:
        ok &= panel.target_dates(h) < np.datetime64(realised_before)
    return np.flatnonzero(ok)


def cv_folds(panel: Panel, h: int, cfg: Config) -> list[Fold]:
    """Expanding-window CV on the development period (everything realised before the test)."""
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    folds = []
    for start, end in cfg["splits"]["cv_blocks"]:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        val = origins_between(panel, h, start, end, realised_before=test_start)
        train = purged_train(panel, h, start)
        if len(val) and len(train):
            folds.append(Fold(f"{start.year}-{end.year - 1}", train, val))
    return folds


def refit_dates(panel: Panel, cfg: Config) -> list[tuple[pd.Timestamp, str]]:
    scfg = cfg["splits"]
    out = [(pd.Timestamp(d), "calibration") for d in scfg["calibration_refits"]]
    test_start = pd.Timestamp(scfg["test_start"])
    d = test_start
    while d <= panel.dates[-1]:
        out.append((d, "test"))
        d = d + pd.DateOffset(years=scfg["test_refit_every_years"])
    return out


def refit_blocks(panel: Panel, h: int, cfg: Config) -> list[RefitBlock]:
    """Walk-forward schedule: at each refit date train on everything realised before it and
    predict the origins until the next refit date."""
    dates = refit_dates(panel, cfg)
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    blocks = []
    for i, (start, phase) in enumerate(dates):
        last = panel.dates[-1] + pd.Timedelta(days=1)
        end = dates[i + 1][0] if i + 1 < len(dates) else last
        if phase == "calibration":
            end = min(end, test_start)
        predict = origins_between(panel, h, start, end)
        train = purged_train(panel, h, start)
        if len(predict) and len(train):
            blocks.append(RefitBlock(start, end, phase, train, predict))
    return blocks
