"""Ronda 4: relación oro/crudo (docs/experimento-ronda4.md).

R4 combina la red W1 a 5 sesiones con las redes nuevas a 63 y 252. La decisión usa la misma
regla y el mismo código que la ronda 3 (:mod:`oilcast.evaluation.ronda3`). El test incluye
además una referencia lineal recursiva con la misma variable, que no es candidata.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from oilcast.config import Config, load_config
from oilcast.evaluation.ronda3 import Round, decide, test_report
from oilcast.evaluation.summary import ensemble, frame_metrics, recalibrate_center
from oilcast.features.windows import Panel
from oilcast.paths import PROJECT_ROOT, Paths, get_paths
from oilcast.training.evaluate import _frame, naive_predictions, quantile_columns
from oilcast.training.multitask import schedule
from oilcast.training.splits import RefitBlock, origins_between, purged_train

OUT = Path("reports/experiments/ronda4")
W1_DIR = "reports/experiments/ronda3/w1"
R4_DIR = "reports/experiments/ronda4/oro"
LINEAR_DIR = "reports/experiments/ronda4/lineal"
RULE = (
    "Regla fijada el 20/11/2026 antes de la búsqueda (docs/experimento-ronda4.md): se elige la "
    "combinación de variante y red con mayor mejora media del MAE frente al ingenuo en los tres "
    "horizontes, en orígenes de 2008 a 2021 con objetivo anterior a 2022. Sustituye a la RNN W1 "
    "solo si la supera en 2008-2021 y no es peor que ella en 2008-2013."
)
RONDA4 = Round(
    OUT,
    {"W1": W1_DIR, "R4": R4_DIR},
    ("W1", "rnn"),
    RULE,
    ("docs/experimento-ronda4.md", "configs/ronda4.yaml"),
)


def _base_cfg() -> Config:
    """Configuración de W1 (tres horizontes) para puntuar todas las variantes por igual."""
    return copy.deepcopy(load_config(PROJECT_ROOT / "configs/ronda3_w1.yaml"))


def assemble(cfg: Config) -> None:
    """R4 a 5 sesiones = red W1 publicada (mismas predicciones, copiadas)."""
    src, dst = get_paths({"paths": {"results_dir": W1_DIR}}), get_paths(cfg)
    for arch in cfg["architectures"]:
        target = dst.predictions_csv(arch, 5)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src.predictions_csv(arch, 5), target)


def decide4(cfg: Config) -> dict:
    assemble(cfg)
    return decide(_base_cfg(), RONDA4)


def linear_reference(cfg: Config, paths: Paths) -> pd.DataFrame:
    """Regresión lineal recursiva sobre ``oro_crudo`` con los mismos reentrenos (sin deriva)."""
    panel = Panel.load(paths, cfg)
    x = panel.features[:, panel.feature_names.index("oro_crudo")].astype(float)
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    out = get_paths({"paths": {"results_dir": LINEAR_DIR}})
    for h in cfg["targets"]["horizons"]:
        y = panel.target(h)
        frames = []
        for start, end, phase in schedule(panel, cfg):
            train = purged_train(panel, h, start)
            block = RefitBlock(start, end, phase, train, origins_between(panel, h, start, end))
            xc = x[train] - x[train].mean()
            beta = float(xc @ (y[train] - y[train].mean()) / (xc @ xc))
            p = beta * (x[block.predict] - x[train].mean())
            df = _frame(panel, h, block, np.tile(p[:, None], (1, len(qcols))), qcols)
            df.insert(4, "seed", 0)
            frames.append(df)
        path = out.predictions_csv("lineal", h)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format="%.6g")
    rows = []
    base = _base_cfg()
    qcols = quantile_columns(base["targets"]["quantiles"])
    for h in cfg["targets"]["horizons"]:
        pred = pd.read_csv(
            out.predictions_csv("lineal", h), parse_dates=["origin", "target_date", "refit_date"]
        )
        ens = recalibrate_center(ensemble(pred, qcols), base)
        naive = naive_predictions(base, panel, h)
        test = ens[ens["phase"] == "test"]
        m = frame_metrics(test, naive[naive["phase"] == "test"], h, base, calibrated=False)
        rows.append(
            {"variant": "lineal", "arch": "lineal", "horizon": h, **{k: m[k] for k in METRICS}}
        )
    return pd.DataFrame(rows)


METRICS = ("mae", "rmse", "skill_mae", "skill_rmse", "dm_pvalue", "dir_acc", "n_test")


def test4(cfg: Config, paths: Paths) -> pd.DataFrame:
    test = test_report(_base_cfg(), paths, RONDA4)
    lin = linear_reference(cfg, paths)
    lin.to_csv(PROJECT_ROOT / OUT / "test_lineal.csv", index=False)
    return test
