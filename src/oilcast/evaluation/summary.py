"""Turn walk-forward predictions into metrics tables, the model choice and conformal state.

Everything here is computed on the *test* phase only (origins from ``splits.test_start``),
on exactly the same origins for the networks and the naive baseline. Point forecasts are
the median quantile converted to price: ``P[t] * exp(q50)``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from oilcast.config import Config, for_horizon
from oilcast.evaluation.conformal import (
    conformal_offset,
    cqr_scores,
    online_offsets,
    online_shrink,
)
from oilcast.evaluation.metrics import (
    diebold_mariano,
    directional_accuracy,
    interval_metrics,
    pinball,
    point_metrics,
)
from oilcast.features.windows import Panel
from oilcast.paths import Paths
from oilcast.training.evaluate import naive_predictions, quantile_columns
from oilcast.utils import write_json

log = logging.getLogger(__name__)

SELECTION_RULE = (
    "se elige la red con menor MAE relativo medio (MAE del modelo / MAE del ingenuo, media de "
    "los tres horizontes, ensemble de semillas). Si las dos mejores están a menos de 1 punto "
    "porcentual, se prefiere la más estable entre semillas y, después, la más barata (menos "
    "parámetros)."
)

KEYS = ["origin", "target_date", "refit_date", "phase", "price", "y"]


def ensemble(pred: pd.DataFrame, qcols: list[str]) -> pd.DataFrame:
    """Average the (sorted) quantiles of all seeds for each origin."""
    ens = pred.groupby(KEYS, sort=True)[qcols].mean().reset_index()
    ens[qcols] = np.sort(ens[qcols].to_numpy(), axis=1)
    return ens


def recalibrate_center(ens: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Shrink the central forecast towards the naive one (``evaluation.recalibrate_center``).

    The whole predictive distribution is shifted so its median becomes ``lam * q50``, with
    ``lam`` estimated at each origin from outcomes already known (see ``online_shrink``).
    ``q50_raw`` keeps the network's own median, used to estimate ``lam``.
    """
    out = ens.copy()
    out["q50_raw"] = out["q50"]
    if not cfg["evaluation"].get("recalibrate_center"):
        out["lam"] = 1.0
        return out
    lam = online_shrink(
        out["origin"].to_numpy(),
        out["target_date"].to_numpy(),
        out["q50"].to_numpy(),
        out["y"].to_numpy(),
        cfg["evaluation"]["center_min_scores"],
    )
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    shift = (lam - 1.0) * out["q50"].to_numpy()
    out[qcols] = out[qcols].to_numpy() + shift[:, None]
    out["lam"] = lam
    return out


def latest_shrink(ens: pd.DataFrame, cfg: Config, asof: pd.Timestamp) -> float:
    """Shrinkage factor for a forecast issued at ``asof`` (production use)."""
    if not cfg["evaluation"].get("recalibrate_center"):
        return 1.0
    lam = online_shrink(
        np.array([np.datetime64(asof)]),
        ens["target_date"].to_numpy(),
        ens["q50_raw"].to_numpy(),
        ens["y"].to_numpy(),
        cfg["evaluation"]["center_min_scores"],
    )
    return float(lam[0])


def add_conformal(ens: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add calibrated interval bounds ``lo{L}``/``hi{L}`` (log-return space) per level L."""
    ecfg = cfg["evaluation"]
    out = ens.copy()
    origins = out["origin"].to_numpy()
    tds = out["target_date"].to_numpy()
    for level, (ql, qh) in cfg["targets"]["intervals"].items():
        lo, hi = out[quantile_columns([ql])[0]], out[quantile_columns([qh])[0]]
        scores = cqr_scores(out["y"].to_numpy(), lo.to_numpy(), hi.to_numpy())
        off = online_offsets(
            origins,
            tds,
            scores,
            int(level) / 100,
            ecfg["conformal_window"],
            ecfg["conformal_min_scores"],
        )
        out[f"offset{level}"] = off
        out[f"lo{level}"] = lo - off
        out[f"hi{level}"] = hi + off
    return out


def latest_offsets(ens: pd.DataFrame, cfg: Config, asof: pd.Timestamp) -> dict[str, float]:
    """Conformal offsets for a forecast issued at ``asof`` (production use)."""
    ecfg = cfg["evaluation"]
    out = {}
    for level, (ql, qh) in cfg["targets"]["intervals"].items():
        lo, hi = ens[quantile_columns([ql])[0]], ens[quantile_columns([qh])[0]]
        scores = pd.Series(
            cqr_scores(ens["y"].to_numpy(), lo.to_numpy(), hi.to_numpy()), index=ens["target_date"]
        ).sort_index()
        past = scores[scores.index < asof].to_numpy()
        if ecfg["conformal_window"]:
            past = past[-ecfg["conformal_window"] :]
        out[level] = conformal_offset(past, int(level) / 100) if len(past) else 0.0
    return out


def frame_metrics(
    df: pd.DataFrame, naive: pd.DataFrame, h: int, cfg: Config, calibrated: bool
) -> dict[str, float]:
    """Metrics of one forecast frame against the naive frame on the same origins."""
    quantiles = cfg["targets"]["quantiles"]
    qcols = quantile_columns(quantiles)
    df = df.set_index("origin").sort_index()
    nv = naive.set_index("origin").loc[df.index]
    price = df["price"].to_numpy()
    actual = price * np.exp(df["y"].to_numpy())
    pred = price * np.exp(df["q50"].to_numpy())
    m = point_metrics(actual, pred)
    m_naive = point_metrics(actual, price)
    m["skill_mae"] = 1 - m["mae"] / m_naive["mae"]
    m["skill_rmse"] = 1 - m["rmse"] / m_naive["rmse"]
    m["dir_acc"] = directional_accuracy(price, actual, pred)
    m.update(diebold_mariano(pred - actual, price - actual, h, cfg["evaluation"]["dm_loss"]))
    m["pinball"] = pinball(df["y"].to_numpy(), df[qcols].to_numpy(), quantiles)
    m["pinball_naive"] = pinball(nv["y"].to_numpy(), nv[qcols].to_numpy(), quantiles)
    for level, (ql, qh) in cfg["targets"]["intervals"].items():
        if calibrated:
            lo, hi = df[f"lo{level}"], df[f"hi{level}"]
        else:
            lo, hi = df[quantile_columns([ql])[0]], df[quantile_columns([qh])[0]]
        im = interval_metrics(
            actual, price * np.exp(lo.to_numpy()), price * np.exp(hi.to_numpy()), int(level) / 100
        )
        m.update({f"{k}{level}": v for k, v in im.items()})
    m["up_share"] = float(np.mean(actual > price))  # reference for directional accuracy
    m["n_test"] = len(df)
    m["first_origin"] = df.index.min().date().isoformat()
    m["last_origin"] = df.index.max().date().isoformat()
    return m


def naive_metrics(naive: pd.DataFrame, h: int, cfg: Config) -> dict[str, float]:
    m = frame_metrics(naive, naive, h, cfg, calibrated=False)
    m.update({"skill_mae": 0.0, "skill_rmse": 0.0, "dm_stat": np.nan, "dm_pvalue": np.nan})
    return m


def by_year(df: pd.DataFrame, naive: pd.DataFrame) -> pd.DataFrame:
    """MAE of model and naive per calendar year of the forecast origin (test phase)."""
    d = df.set_index("origin").sort_index()
    nv = naive.set_index("origin").loc[d.index]
    actual = d["price"] * np.exp(d["y"])
    err_m = (d["price"] * np.exp(d["q50"]) - actual).abs()
    err_n = (nv["price"] - actual).abs()
    out = pd.DataFrame({"mae": err_m, "mae_naive": err_n}).groupby(d.index.year).mean()
    out["skill_mae"] = 1 - out["mae"] / out["mae_naive"]
    out["n"] = d.groupby(d.index.year).size()
    out.index.name = "year"
    return out.reset_index()


def choose_architecture(comp: pd.DataFrame, archs: list[str]) -> dict:
    """Pick the production architecture.

    Rule: lowest mean relative MAE (MAE / naive MAE) across the three horizons for the
    seed ensemble. If the best two are within 1 percentage point, prefer the more stable one
    (lower mean seed std of MAE relative to its MAE), then the cheaper one (fewer params).
    """
    rows = []
    for arch in archs:
        c = comp[comp["model"] == arch]
        rows.append(
            {
                "arch": arch,
                "mean_rel_mae": float((1 - c["skill_mae"]).mean()),
                "mean_rel_seed_std": float((c["mae_seed_std"] / c["mae"]).mean()),
                "mean_params": float(c["n_params"].mean())
                if c["n_params"].notna().all()
                else np.inf,
                "mean_coverage80": float(c["coverage80"].mean()),
            }
        )
    score = pd.DataFrame(rows).sort_values("mean_rel_mae").reset_index(drop=True)
    contenders = score[score["mean_rel_mae"] <= score["mean_rel_mae"].iloc[0] + 0.01]
    chosen = contenders.sort_values(["mean_rel_seed_std", "mean_params"]).iloc[0]["arch"]
    return {
        "chosen": chosen,
        "rule": SELECTION_RULE,
        "scores": score.to_dict(orient="records"),
    }


def summarize(cfg: Config, paths: Paths) -> dict:
    """Compute every table from the saved predictions; write them to the results dir."""
    archs, horizons = cfg["architectures"], cfg["targets"]["horizons"]
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    panel = Panel.load(paths, cfg)
    asof = panel.dates[-1] + pd.Timedelta(days=1)
    timing_path = paths.results / "timing.csv"
    timing = (
        pd.read_csv(timing_path)
        if timing_path.exists()
        else pd.DataFrame(columns=["arch", "horizon", "train_seconds", "n_params"])
    )

    comp_rows, seed_rows, year_frames, offsets, shrink = [], [], [], {}, {}
    pred_dir = paths.results / "predictions"
    for h in horizons:
        naive = naive_predictions(cfg, Panel.load(paths, for_horizon(cfg, h)), h)
        naive.to_csv(pred_dir / f"naive_h{h}.csv", index=False, float_format="%.6g")
        naive_test = naive[naive["phase"] == "test"]
        comp_rows.append({"model": "naive", "horizon": h, **naive_metrics(naive_test, h, cfg)})
        for arch in archs:
            path = paths.predictions_csv(arch, h)
            if not path.exists():
                log.warning("missing predictions %s", path)
                continue
            pred = pd.read_csv(path, parse_dates=["origin", "target_date", "refit_date"])
            ens = add_conformal(recalibrate_center(ensemble(pred, qcols), cfg), cfg)
            ens.to_csv(pred_dir / f"{arch}_h{h}_ensemble.csv", index=False, float_format="%.6g")
            offsets.setdefault(arch, {})[str(h)] = latest_offsets(ens, cfg, asof)
            shrink.setdefault(arch, {})[str(h)] = latest_shrink(ens, cfg, asof)
            test = ens[ens["phase"] == "test"]
            m = frame_metrics(test, naive_test, h, cfg, calibrated=True)
            raw = frame_metrics(test, naive_test, h, cfg, calibrated=False)
            m.update({f"raw_{k}": raw[k] for k in ("coverage80", "width80", "coverage50")})
            seed_m = []
            for seed, g in pred[pred["phase"] == "test"].groupby("seed"):
                sm = frame_metrics(g.drop(columns=["seed"]), naive_test, h, cfg, calibrated=False)
                seed_rows.append({"model": arch, "horizon": h, "seed": seed, **sm})
                seed_m.append(sm)
            m["mae_seed_mean"] = float(np.mean([s["mae"] for s in seed_m]))
            m["mae_seed_std"] = float(np.std([s["mae"] for s in seed_m]))
            m["rmse_seed_std"] = float(np.std([s["rmse"] for s in seed_m]))
            m["skill_mae_seed_std"] = float(np.std([s["skill_mae"] for s in seed_m]))
            t = timing[(timing["arch"] == arch) & (timing["horizon"] == h)]
            m["n_params"] = int(t["n_params"].iloc[0]) if len(t) else np.nan
            m["train_seconds"] = float(t["train_seconds"].iloc[0]) if len(t) else np.nan
            comp_rows.append({"model": arch, "horizon": h, **m})
            yb = by_year(test, naive_test)
            yb.insert(0, "horizon", h)
            yb.insert(0, "model", arch)
            year_frames.append(yb)

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(paths.results / "comparison.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(paths.results / "metrics_by_seed.csv", index=False)
    if year_frames:
        pd.concat(year_frames).to_csv(paths.results / "metrics_by_year.csv", index=False)
    selection = choose_architecture(comp[comp["model"] != "naive"], archs)
    fixed = cfg.get("selection") or {}
    if fixed.get("arch") in archs:  # choice fixed beforehand (docs/experimento-ronda2.md)
        selection.update(chosen=fixed["arch"], rule=fixed["rule"])
        if fixed.get("alternatives"):  # variantes que ganan a la elegida en algún plazo del test
            selection["alternatives"] = fixed["alternatives"]
            selection["alternatives_note"] = fixed.get("alternatives_note", "")
    selection["conformal_offsets"] = offsets
    selection["center_shrink"] = shrink
    selection["offsets_asof"] = asof.date().isoformat()
    write_json(paths.results / "selection.json", selection)
    log.info("chosen architecture: %s", selection["chosen"])
    return selection
