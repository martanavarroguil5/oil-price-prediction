"""Hyperparameter search with Optuna (TPE + median pruning across folds).

Every (architecture, horizon) study uses the same search space, the same number of trials
and the same sampler seed, so the three networks get an equal tuning budget.
Objective: mean validation pinball loss over the purged walk-forward folds.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
import torch

from oilcast.config import Config
from oilcast.features.windows import Panel
from oilcast.models.recurrent import HParams
from oilcast.paths import Paths
from oilcast.training.splits import cv_folds
from oilcast.training.trainer import fit, prepare_data
from oilcast.utils import get_device, read_json, write_json

log = logging.getLogger(__name__)


def suggest_hparams(trial: optuna.Trial, space: dict) -> HParams:
    return HParams(
        hidden_size=trial.suggest_categorical("hidden_size", space["hidden_size"]),
        num_layers=trial.suggest_int("num_layers", *space["num_layers"]),
        dropout=trial.suggest_float("dropout", *space["dropout"]),
        lr=trial.suggest_float("lr", *space["lr"], log=True),
        weight_decay=trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        batch_size=trial.suggest_categorical("batch_size", space["batch_size"]),
    )


def run_study(cfg: Config, paths: Paths, arch: str, h: int) -> dict:
    """Run (or resume) the study for one architecture and horizon; writes ``best.json``."""
    torch.set_num_threads(cfg["training"]["num_threads"])
    device = get_device()
    panel = Panel.load(paths, cfg)
    folds = cv_folds(panel, h, cfg)
    prepared = [prepare_data(panel, h, f.train, cfg, device) for f in folds]
    for f in folds:
        log.info("%s h=%d fold %s: train=%d val=%d", arch, h, f.name, len(f.train), len(f.val))

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hparams(trial, cfg["tuning"]["space"])
        losses, best_epochs = [], []
        for k, (fold, data) in enumerate(zip(folds, prepared, strict=True)):
            res = fit(
                arch,
                hp,
                data,
                fold.train,
                cfg,
                epochs=cfg["training"]["max_epochs"],
                seed=cfg["seed"],
                val_idx=fold.val,
            )
            losses.append(res.best_val)
            best_epochs.append(res.best_epoch)
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_losses", losses)
        trial.set_user_attr("best_epochs", best_epochs)
        return float(np.mean(losses))

    out_dir = paths.tuning_dir(arch, h)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["tuning"]
    study = optuna.create_study(
        study_name=f"{arch}_h{h}",
        storage=f"sqlite:///{out_dir / 'study.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=cfg["seed"], n_startup_trials=tcfg["startup_trials"]
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=tcfg["pruner_startup_trials"]),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, tcfg["n_trials"] - done))

    best = study.best_trial
    best_epochs = best.user_attrs["best_epochs"]
    result = {
        "arch": arch,
        "horizon": h,
        "params": best.params,
        "value": best.value,
        "fold_names": [f.name for f in folds],
        "fold_losses": best.user_attrs["fold_losses"],
        "best_epochs": best_epochs,
        # refits train exactly the median best epoch found in the CV folds
        "final_epochs": max(1, round(float(np.median(best_epochs)))),
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "device": str(device),
    }
    write_json(out_dir / "best.json", result)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    log.info("%s h=%d best=%.4f params=%s", arch, h, best.value, best.params)
    return result


def load_best(paths: Paths, arch: str, h: int) -> tuple[HParams, int]:
    """Tuned hyperparameters and the fixed epoch count for refits."""
    best = read_json(paths.tuning_dir(arch, h) / "best.json")
    return HParams(**best["params"]), int(best["final_epochs"])


def tuning_summary(paths: Paths, archs: list[str], horizons: list[int]) -> pd.DataFrame:
    rows = []
    for arch in archs:
        for h in horizons:
            f = paths.tuning_dir(arch, h) / "best.json"
            if f.exists():
                b = read_json(f)
                rows.append(
                    {
                        "arch": arch,
                        "horizon": h,
                        "cv_pinball": b["value"],
                        "final_epochs": b["final_epochs"],
                        "n_trials": b["n_trials"],
                        "n_pruned": b["n_pruned"],
                        **b["params"],
                    }
                )
    return pd.DataFrame(rows)
