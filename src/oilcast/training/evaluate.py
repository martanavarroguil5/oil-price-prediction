"""Walk-forward out-of-sample evaluation with periodic refits.

At every refit date the network is retrained from scratch on all samples whose target is
realised before that date, with the tuned hyperparameters and the *fixed* epoch count from
CV (the test period is never used for early stopping), for several seeds. It then predicts
every origin until the next refit date. Pre-test refits produce out-of-sample predictions
that are used only to calibrate the conformal interval.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from oilcast.config import Config
from oilcast.features.windows import Panel
from oilcast.models.naive import naive_quantiles
from oilcast.paths import Paths
from oilcast.training.splits import RefitBlock, refit_blocks
from oilcast.training.trainer import fit, predict_scaled, prepare_data
from oilcast.training.tune import load_best
from oilcast.utils import get_device

log = logging.getLogger(__name__)


def quantile_columns(quantiles: list[float]) -> list[str]:
    return [f"q{round(q * 100):02d}" for q in quantiles]


def _frame(
    panel: Panel, h: int, block: RefitBlock, q: np.ndarray, qcols: list[str]
) -> pd.DataFrame:
    idx = block.predict
    df = pd.DataFrame(np.sort(q, axis=1), columns=qcols)
    df.insert(0, "origin", panel.dates[idx])
    df.insert(1, "target_date", pd.DatetimeIndex(panel.target_dates(h)[idx]))
    df.insert(2, "refit_date", block.refit_date)
    df.insert(3, "phase", block.phase)
    df.insert(4, "price", panel.price[idx])
    df.insert(5, "y", panel.target(h)[idx])
    return df


def run_evaluation(cfg: Config, paths: Paths, arch: str, h: int) -> pd.DataFrame:
    """Predict every refit block with every seed; writes ``predictions/{arch}_h{h}.csv``."""
    torch.set_num_threads(cfg["training"]["num_threads"])
    device = get_device()
    panel = Panel.load(paths, cfg)
    hp, epochs = load_best(paths, arch, h)
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    frames = []
    for block in refit_blocks(panel, h, cfg):
        data = prepare_data(panel, h, block.train, cfg, device)
        for seed in cfg["evaluation"]["seeds"]:
            res = fit(arch, hp, data, block.train, cfg, epochs=epochs, seed=seed)
            q = predict_scaled(res.model, data, block.predict) * data.y_scale
            df = _frame(panel, h, block, q, qcols)
            df.insert(4, "seed", seed)
            df["train_seconds"] = res.seconds
            frames.append(df)
        log.info(
            "%s h=%d refit %s (%s): train=%d predict=%d",
            arch,
            h,
            block.refit_date.date(),
            block.phase,
            len(block.train),
            len(block.predict),
        )
    out = pd.concat(frames, ignore_index=True)
    path = paths.predictions_csv(arch, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.6g")
    return out


def naive_predictions(cfg: Config, panel: Panel, h: int) -> pd.DataFrame:
    """Naive baseline on exactly the same refit blocks and origins as the networks."""
    quantiles = cfg["targets"]["quantiles"]
    qcols = quantile_columns(quantiles)
    y = panel.target(h)
    frames = []
    for block in refit_blocks(panel, h, cfg):
        q = naive_quantiles(y[block.train], quantiles)
        df = _frame(panel, h, block, np.tile(q, (len(block.predict), 1)), qcols)
        df.insert(4, "seed", -1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
