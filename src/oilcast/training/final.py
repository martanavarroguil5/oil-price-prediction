"""Production fit: the chosen architecture, one seed ensemble per horizon, all data.

The *spec* (architecture, tuned hyperparameters, epochs, conformal offsets and the test
metrics shown in the app) is built from the evaluation results the first time and then
stored in ``models/metadata.json``, so the app can retrain later without the results dir.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch

from oilcast.config import Config
from oilcast.features.windows import Panel
from oilcast.models.recurrent import ARCH_LABELS, HParams, count_parameters
from oilcast.paths import Paths
from oilcast.training.trainer import fit, prepare_data
from oilcast.training.tune import load_best
from oilcast.utils import get_device, read_json, write_json

log = logging.getLogger(__name__)

SUMMARY_COLS = [
    "mae",
    "mse",
    "r2",
    "rmse",
    "skill_mae",
    "skill_rmse",
    "dm_pvalue",
    "dir_acc",
    "coverage80",
    "width80",
    "n_test",
    "first_origin",
    "last_origin",
]


def _py(v):
    """Plain Python scalar (JSON-friendly) from a pandas/numpy value."""
    return v.item() if hasattr(v, "item") else v


def spec_from_results(cfg: Config, paths: Paths, arch: str | None = None) -> dict:
    selection = read_json(paths.results / "selection.json")
    arch = arch or (
        cfg["final"]["architecture"]
        if cfg["final"]["architecture"] != "auto"
        else selection["chosen"]
    )
    comp = pd.read_csv(paths.results / "comparison.csv")
    by_year_path = paths.results / "metrics_by_year.csv"
    by_year = pd.read_csv(by_year_path) if by_year_path.exists() else pd.DataFrame()
    horizons = {}
    for h in cfg["targets"]["horizons"]:
        hp, epochs = load_best(paths, arch, h)
        row = comp[(comp["model"] == arch) & (comp["horizon"] == h)].iloc[0]
        naive = comp[(comp["model"] == "naive") & (comp["horizon"] == h)].iloc[0]
        yb = by_year[(by_year["model"] == arch) & (by_year["horizon"] == h)] if len(by_year) else []
        horizons[str(h)] = {
            "hparams": hp.to_dict(),
            "epochs": epochs,
            "conformal_offsets": selection["conformal_offsets"][arch][str(h)],
            "center_shrink": selection.get("center_shrink", {}).get(arch, {}).get(str(h), 1.0),
            "test_metrics": {k: _py(row[k]) for k in SUMMARY_COLS if k in row},
            "naive_metrics": {
                k: _py(naive[k])
                for k in ("mae", "rmse", "mse", "r2", "coverage80", "width80")
                if k in naive
            },
            "skill_by_year": {int(r.year): float(r.skill_mae) for r in yb.itertuples()}
            if len(yb)
            else {},
        }
    return {
        "arch": arch,
        "arch_label": ARCH_LABELS[arch],
        "selection_rule": selection["rule"],
        "remove_drift": bool(cfg["targets"].get("remove_drift", False)),
        "horizons": horizons,
    }


def spec_from_metadata(paths: Paths) -> dict:
    return read_json(paths.models / "metadata.json")["spec"]


def train_final(
    cfg: Config,
    paths: Paths,
    spec: dict,
    seeds: list[int] | None = None,
    device: torch.device | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Fit and save the production ensembles; returns the new metadata."""
    seeds = seeds if seeds is not None else cfg["final"]["seeds"]
    if spec.get("remove_drift"):  # train the way the spec was evaluated
        cfg = {**cfg, "targets": {**cfg["targets"], "remove_drift": True}}
    device = device or get_device()
    panel = Panel.load(paths, cfg)
    arch = spec["arch"]
    paths.models.mkdir(parents=True, exist_ok=True)
    total = len(spec["horizons"]) * len(seeds)
    done = 0
    info = {}
    for hs, hspec in spec["horizons"].items():
        h = int(hs)
        hp, epochs = HParams(**hspec["hparams"]), int(hspec["epochs"])
        train = np.flatnonzero(panel.usable(h))
        data = prepare_data(panel, h, train, cfg, device)
        states, seconds = [], 0.0
        for seed in seeds:
            if progress:
                progress(done / total, f"Horizonte {h} días · semilla {seed}")
            res = fit(arch, hp, data, train, cfg, epochs=epochs, seed=seed)
            states.append({k: v.detach().cpu() for k, v in res.model.state_dict().items()})
            seconds += res.seconds
            done += 1
        torch.save(
            {
                "arch": arch,
                "hparams": hp.to_dict(),
                "states": states,
                "scaler": data.scaler.to_dict(),
                "y_scale": data.y_scale,
            },
            paths.models / f"h{h}.pt",
        )
        info[hs] = {
            "n_train": len(train),
            "last_train_origin": panel.dates[train[-1]].date(),
            "train_seconds": seconds,
            "n_params": count_parameters(res.model),
        }
        log.info("final %s h=%d: %d seeds, %.1fs", arch, h, len(seeds), seconds)
    if progress:
        progress(1.0, "Modelo guardado")
    meta = {
        "arch": arch,
        "arch_label": ARCH_LABELS[arch],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_data_date": panel.dates[-1].date(),
        "device": "CPU" if device.type == "cpu" else "GPU",
        "seeds": seeds,
        "window": panel.window,
        "feature_names": panel.feature_names,
        "quantiles": cfg["targets"]["quantiles"],
        "intervals": cfg["targets"]["intervals"],
        "fit": info,
        "spec": spec,
    }
    write_json(paths.models / "metadata.json", meta)
    return meta
