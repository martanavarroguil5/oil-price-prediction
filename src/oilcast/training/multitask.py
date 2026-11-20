"""Ronda 3: una red por arquitectura que aprende a la vez los tres horizontes.

Cada origen aporta hasta tres objetivos (5, 63 y 252 sesiones). Una misma red recurrente
(RNN simple, LSTM o GRU) con una salida por horizonte aprende los cuantiles de los tres.
La purga se aplica por horizonte: un objetivo entra en el entrenamiento solo si ya se había
realizado antes del corte, y los demás objetivos del mismo origen se enmascaran. Ninguna
pérdida usa un objetivo posterior al corte.

Protocolo y regla de decisión: ``docs/experimento-ronda3.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import torch

from oilcast.config import Config
from oilcast.data.dataset import load_dataset
from oilcast.evaluation.summary import ensemble, recalibrate_center
from oilcast.features.windows import Panel, Scaler, scaler_rows
from oilcast.models.recurrent import HParams, build_model, count_parameters
from oilcast.paths import Paths, get_paths
from oilcast.training.evaluate import _frame, quantile_columns
from oilcast.training.splits import RefitBlock, origins_between, purged_train, refit_dates
from oilcast.training.trainer import fit, predict_scaled, prepare_data
from oilcast.training.tune import load_best, suggest_hparams
from oilcast.utils import get_device, read_json, write_json

log = logging.getLogger(__name__)


def load_panel(cfg: Config, paths: Paths) -> Panel:
    """Main dataset, plus refined-product prices when ``features.products`` is set."""
    df = load_dataset(paths)
    if cfg["features"].get("products"):
        from oilcast.data.extra import load_products

        df = df.join(load_products(paths, df.index))
    return Panel.from_dataset(df, cfg)


def targets(panel: Panel, horizons: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(T, K) log-return targets and their realisation dates."""
    y = np.stack([panel.target(h) for h in horizons], axis=1)
    td = np.stack([panel.target_dates(h) for h in horizons], axis=1)
    return y, td


def known_before(panel: Panel, horizons: list[int], cutoff) -> np.ndarray:
    """(T, K) True where the target exists and was realised strictly before ``cutoff``."""
    y, td = targets(panel, horizons)
    return panel.valid[:, None] & np.isfinite(y) & (td < np.datetime64(pd.Timestamp(cutoff)))


@dataclass
class MultiData:
    """Device tensors for one training set; ``y`` is (T, K) with NaN where masked."""

    features: torch.Tensor
    y: torch.Tensor
    scaler: Scaler
    scales: np.ndarray
    centers: np.ndarray
    window: int

    @property
    def device(self) -> torch.device:
        return self.features.device


def prepare(
    panel: Panel,
    horizons: list[int],
    rows: np.ndarray,
    known: np.ndarray,
    cfg: Config,
    device: torch.device,
    val: tuple[np.ndarray, np.ndarray] | None = None,
) -> MultiData:
    """Scaler and per-horizon target scale fitted on the known training targets only."""
    scaler = Scaler.fit(panel.features, scaler_rows(rows), tuple(cfg["features"]["clip_quantiles"]))
    y, _ = targets(panel, horizons)
    k = len(horizons)
    scales, centers = np.ones(k), np.zeros(k)
    for j in range(k):
        yj = y[rows[known[:, j]], j]
        scales[j] = float(np.std(yj))
        if cfg["targets"].get("remove_drift"):
            centers[j] = float(np.mean(yj))
    z = (y - centers) / scales
    out = np.full(y.shape, np.nan, dtype=np.float32)
    out[rows] = np.where(known, z[rows], np.nan)
    if val is not None:
        vrows, vknown = val
        out[vrows] = np.where(vknown, z[vrows], np.nan)
    return MultiData(
        features=torch.from_numpy(scaler.transform(panel.features)).to(device),
        y=torch.from_numpy(out).to(device),
        scaler=scaler,
        scales=scales,
        centers=centers,
        window=panel.window,
    )


def multi_pinball(k: int, n_q: int):
    """Pinball loss averaged per horizon over its available targets, then across horizons."""

    def objective(model, x, target, q):
        pred = model(x).view(len(x), k, n_q)
        mask = torch.isfinite(target)
        diff = torch.nan_to_num(target)[..., None] - pred
        loss = torch.maximum(q * diff, (q - 1) * diff).mean(-1) * mask
        count = mask.sum(0)
        present = (count > 0).float()
        per_h = loss.sum(0) / count.clamp(min=1)
        return (per_h * present).sum() / present.sum().clamp(min=1)

    return objective


def median_score(k: int, n_q: int, q50: int):
    """Validation score aligned with the MAE: error of the median relative to the naive
    forecast (log returns), per horizon, averaged across horizons. Lower is better."""

    def score(model, data: MultiData, idx: np.ndarray) -> float:
        pred = predict_scaled(model, data, idx).reshape(len(idx), k, n_q)[:, :, q50]
        z = data.y[torch.as_tensor(idx, device=data.device)].cpu().numpy()
        # The forecast is scale * output: the training drift is NOT added back, so compare
        # against the uncentred target y / scale; the naive forecast is 0.
        actual = z + data.centers / data.scales
        rel = []
        for j in range(k):
            ok = np.isfinite(actual[:, j])
            if ok.any():
                err = np.abs(actual[ok, j] - pred[ok, j]).mean()
                rel.append(err / np.abs(actual[ok, j]).mean())
        return float(np.mean(rel))

    return score


def validation_score(cfg: Config, k: int):
    """``tuning.objective``: 'pinball' (all quantiles, as W1) or 'median'."""
    quantiles = cfg["targets"]["quantiles"]
    if cfg["tuning"].get("objective", "pinball") == "median":
        return median_score(k, len(quantiles), quantiles.index(0.5))
    return None


def factory(k: int):
    return lambda arch, n_features, hp, n_q: build_model(arch, n_features, hp, n_q * k)


def train_set(panel: Panel, horizons: list[int], cutoff) -> tuple[np.ndarray, np.ndarray]:
    known = known_before(panel, horizons, cutoff)
    rows = np.flatnonzero(known.any(axis=1))
    return rows, known[rows]


def cv_folds(panel: Panel, horizons: list[int], cfg: Config) -> list[dict]:
    """Same validation blocks as the single-horizon study; targets realised before the test."""
    test_start = cfg["splits"]["test_start"]
    vknown_all = known_before(panel, horizons, test_start)
    d = panel.dates.values
    folds = []
    for start, end in cfg["splits"]["cv_blocks"]:
        rows, known = train_set(panel, horizons, start)
        in_block = (d >= np.datetime64(pd.Timestamp(start))) & (
            d < np.datetime64(pd.Timestamp(end))
        )
        vrows = np.flatnonzero(in_block & vknown_all.any(axis=1))
        folds.append(
            {
                "name": f"{start[:4]}-{int(end[:4]) - 1}",
                "rows": rows,
                "known": known,
                "vrows": vrows,
                "vknown": vknown_all[vrows],
            }
        )
    return folds


def tuning_dir(paths: Paths, arch: str):
    return paths.results / "tuning" / f"{arch}_multi"


def run_study(cfg: Config, paths: Paths, arch: str) -> dict:
    """Optuna study for one architecture (same space, trials and sampler seed for all)."""
    torch.set_num_threads(cfg["training"]["num_threads"])
    device = get_device()
    panel = load_panel(cfg, paths)
    horizons = cfg["targets"]["horizons"]
    k, n_q = len(horizons), len(cfg["targets"]["quantiles"])
    folds = cv_folds(panel, horizons, cfg)
    prepared = [
        prepare(panel, horizons, f["rows"], f["known"], cfg, device, (f["vrows"], f["vknown"]))
        for f in folds
    ]
    loss_fn = multi_pinball(k, n_q)
    score_fn = validation_score(cfg, k)

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hparams(trial, cfg["tuning"]["space"])
        losses, best_epochs = [], []
        for i, (fold, data) in enumerate(zip(folds, prepared, strict=True)):
            res = fit(
                arch,
                hp,
                data,
                fold["rows"],
                cfg,
                epochs=cfg["training"]["max_epochs"],
                seed=cfg["seed"],
                val_idx=fold["vrows"],
                model_factory=factory(k),
                objective=loss_fn,
                validation_score=score_fn,
            )
            losses.append(res.best_val)
            best_epochs.append(res.best_epoch)
            trial.report(float(np.mean(losses)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_losses", losses)
        trial.set_user_attr("best_epochs", best_epochs)
        return float(np.mean(losses))

    out_dir = tuning_dir(paths, arch)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["tuning"]
    study = optuna.create_study(
        study_name=f"{arch}_multi",
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
    result = {
        "arch": arch,
        "horizons": horizons,
        "params": best.params,
        "value": best.value,
        "fold_names": [f["name"] for f in folds],
        "fold_losses": best.user_attrs["fold_losses"],
        "best_epochs": best.user_attrs["best_epochs"],
        "final_epochs": max(1, round(float(np.median(best.user_attrs["best_epochs"])))),
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "device": str(device),
    }
    write_json(out_dir / "best.json", result)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    log.info("%s multi best=%.4f params=%s", arch, best.value, best.params)
    return result


def schedule(panel: Panel, cfg: Config) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Refit dates with the end of the block each one predicts (as in ``refit_blocks``)."""
    dates = refit_dates(panel, cfg)
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    out = []
    for i, (start, phase) in enumerate(dates):
        end = dates[i + 1][0] if i + 1 < len(dates) else panel.dates[-1] + pd.Timedelta(days=1)
        if phase == "calibration":
            end = min(end, test_start)
        out.append((start, end, phase))
    return out


def run_evaluation(cfg: Config, paths: Paths, arch: str) -> None:
    """Walk-forward predictions of the multi-horizon network, one CSV per horizon."""
    torch.set_num_threads(cfg["training"]["num_threads"])
    device = get_device()
    panel = load_panel(cfg, paths)
    horizons = cfg["targets"]["horizons"]
    quantiles = cfg["targets"]["quantiles"]
    k, n_q = len(horizons), len(quantiles)
    qcols = quantile_columns(quantiles)
    best = read_json(tuning_dir(paths, arch) / "best.json")
    hp, epochs = HParams(**best["params"]), int(best["final_epochs"])
    loss_fn = multi_pinball(k, n_q)
    frames: dict[int, list[pd.DataFrame]] = {h: [] for h in horizons}
    n_params = None
    for start, end, phase in schedule(panel, cfg):
        rows, known = train_set(panel, horizons, start)
        data = prepare(panel, horizons, rows, known, cfg, device)
        idx = {h: origins_between(panel, h, start, end) for h in horizons}
        union = np.unique(np.concatenate(list(idx.values())))
        if not len(union):
            continue
        for seed in cfg["evaluation"]["seeds"]:
            res = fit(
                arch,
                hp,
                data,
                rows,
                cfg,
                epochs=epochs,
                seed=seed,
                model_factory=factory(k),
                objective=loss_fn,
            )
            n_params = count_parameters(res.model)
            out = predict_scaled(res.model, data, union).reshape(len(union), k, n_q)
            out = out * data.scales[None, :, None]  # no center added back (remove_drift)
            pos = {o: i for i, o in enumerate(union)}
            for j, h in enumerate(horizons):
                if not len(idx[h]):
                    continue
                block = RefitBlock(start, end, phase, rows, idx[h])
                q = out[[pos[o] for o in idx[h]], j, :]
                df = _frame(panel, h, block, q, qcols)
                df.insert(4, "seed", seed)
                df["train_seconds"] = res.seconds
                frames[h].append(df)
        log.info(
            "%s multi refit %s (%s): train=%d predict=%d",
            arch,
            start.date(),
            phase,
            len(rows),
            len(union),
        )
    for h in horizons:
        path = paths.predictions_csv(arch, h)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames[h], ignore_index=True).to_csv(path, index=False, float_format="%.6g")
    write_json(
        paths.results / "tuning" / f"{arch}_multi" / "params.json",
        {"n_params": n_params, "epochs": epochs},
    )


def extend_published(cfg: Config, paths: Paths, arch: str, h: int, source: str) -> None:
    """Published single-horizon network (W1): add the early calibration refits that the
    original run did not have, with its published hyperparameters, and keep its predictions
    from the original refits unchanged."""
    torch.set_num_threads(cfg["training"]["num_threads"])
    device = get_device()
    panel = Panel.load(paths, cfg)
    src = get_paths({"paths": {"results_dir": source}})
    old = pd.read_csv(
        src.predictions_csv(arch, h), parse_dates=["origin", "target_date", "refit_date"]
    )
    first_old = old["refit_date"].min()
    hp, epochs = load_best(src, arch, h)
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    frames = []
    for start, end, phase in schedule(panel, cfg):
        if start >= first_old:
            continue
        end = min(end, first_old)
        train = purged_train(panel, h, start)
        block = RefitBlock(start, end, phase, train, origins_between(panel, h, start, end))
        data = prepare_data(panel, h, train, cfg, device)
        for seed in cfg["evaluation"]["seeds"]:
            res = fit(arch, hp, data, train, cfg, epochs=epochs, seed=seed)
            q = predict_scaled(res.model, data, block.predict) * data.y_scale
            df = _frame(panel, h, block, q, qcols)
            df.insert(4, "seed", seed)
            df["train_seconds"] = res.seconds
            frames.append(df)
        log.info(
            "%s h=%d extra refit %s: train=%d predict=%d",
            arch,
            h,
            start.date(),
            len(train),
            len(block.predict),
        )
    old["refit_date"] = old["refit_date"].dt.strftime("%Y-%m-%d")
    new = pd.concat(frames, ignore_index=True)
    new["refit_date"] = pd.to_datetime(new["refit_date"]).dt.strftime("%Y-%m-%d")
    out = pd.concat([new, old], ignore_index=True)
    path = paths.predictions_csv(arch, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.6g")


def development_scores(cfg: Config, variants: dict[str, str]) -> pd.DataFrame:
    """MAE improvement over the naive forecast on development predictions only.

    Development = calibration refits, origins and targets before ``splits.test_start``. The
    centre shrinkage uses every out-of-sample outcome known at each origin. Test rows are
    never read beyond the filter that drops them.
    """
    test_start = np.datetime64(pd.Timestamp(cfg["splits"]["test_start"]))
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    periods = {
        "2008-2013": ("2008-01-01", "2014-01-01"),
        "2014-2021": ("2014-01-01", "2022-01-01"),
        "2008-2021": ("2008-01-01", "2022-01-01"),
    }
    rows = []
    for variant, results_dir in variants.items():
        vp = get_paths({"paths": {"results_dir": results_dir}})
        for arch in cfg["architectures"]:
            for h in cfg["targets"]["horizons"]:
                pred = pd.read_csv(
                    vp.predictions_csv(arch, h), parse_dates=["origin", "target_date", "refit_date"]
                )
                pred = pred[pred["phase"] == "calibration"]
                ens = recalibrate_center(ensemble(pred, qcols), cfg)
                ens = ens[ens["target_date"].values < test_start]
                for name, (a, b) in periods.items():
                    s = ens[(ens["origin"] >= a) & (ens["origin"] < b)]
                    if not len(s):
                        continue
                    actual = s["price"] * np.exp(s["y"])
                    mae_n = float(np.mean(np.abs(actual - s["price"])))
                    mae = float(np.mean(np.abs(actual - s["price"] * np.exp(s["q50"]))))
                    raw = float(np.mean(np.abs(actual - s["price"] * np.exp(s["q50_raw"]))))
                    rows.append(
                        {
                            "variant": variant,
                            "arch": arch,
                            "horizon": h,
                            "period": name,
                            "n": len(s),
                            "mae": mae,
                            "mae_naive": mae_n,
                            "skill_mae": 1 - mae / mae_n,
                            "skill_raw": 1 - raw / mae_n,
                            "mean_lam": float(s["lam"].mean()),
                        }
                    )
    return pd.DataFrame(rows)
