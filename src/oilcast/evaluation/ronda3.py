"""Ronda 3: decisión con desarrollo y, después, un único cálculo del test.

``decide`` lee solo predicciones de calibración (orígenes y objetivos anteriores al test),
aplica la regla de ``docs/experimento-ronda3.md`` y guarda la decisión con los hashes de los
archivos usados. ``test_report`` se niega a ejecutarse sin esa decisión guardada.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oilcast.config import Config
from oilcast.evaluation.metrics import diebold_mariano
from oilcast.evaluation.summary import add_conformal, ensemble, frame_metrics, recalibrate_center
from oilcast.features.windows import Panel
from oilcast.paths import PROJECT_ROOT, Paths, get_paths
from oilcast.training.evaluate import naive_predictions, quantile_columns
from oilcast.training.multitask import development_scores
from oilcast.utils import read_json, write_json

log = logging.getLogger(__name__)

OUT = Path("reports/experiments/ronda3")
VARIANTS = {
    "W1": "reports/experiments/ronda3/w1",
    "S1": "reports/experiments/ronda3/single",
    "M0": "reports/experiments/ronda3/multi_pinball",
    "M1": "reports/experiments/ronda3/multi",
    "M2": "reports/experiments/ronda3/productos",
}
LABELS = {
    "W1": "W1 publicada",
    "S1": "Una red por plazo, selección por la mediana",
    "M0": "Red multihorizonte, selección por pinball",
    "M1": "Red multihorizonte, selección por la mediana",
    "M2": "Red multihorizonte con márgenes de refino",
}
REFERENCE = ("W1", "rnn")
RULE = (
    "Regla fijada el 19/11/2026 antes de entrenar (docs/experimento-ronda3.md): se elige la "
    "combinación de variante y red con mayor mejora media del MAE frente al ingenuo en los "
    "tres horizontes, en orígenes de 2008 a 2021 con objetivo anterior a 2022. Sustituye a la "
    "RNN W1 solo si la supera en 2008-2021 y no es peor que ella en 2008-2013."
)


@dataclass(frozen=True)
class Round:
    """One pre-registered round: variants, reference, rule and the files that fix it."""

    out: Path
    variants: dict[str, str]
    reference: tuple[str, str]
    rule: str
    fixed: tuple[str, ...]  # protocol and configs hashed with the decision
    labels: dict[str, str] = field(default_factory=dict)


RONDA3 = Round(
    OUT, VARIANTS, REFERENCE, RULE, ("docs/experimento-ronda3.md", "configs/ronda3.yaml")
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _available(cfg: Config, rnd: Round = RONDA3) -> dict[str, str]:
    out = {}
    for name, results_dir in rnd.variants.items():
        vp = get_paths({"paths": {"results_dir": results_dir}})
        files = [
            vp.predictions_csv(a, h)
            for a in cfg["architectures"]
            for h in cfg["targets"]["horizons"]
        ]
        if all(f.exists() for f in files):
            out[name] = results_dir
        else:
            log.warning("variant %s incomplete, skipped", name)
    return out


def decide(cfg: Config, rnd: Round = RONDA3) -> dict:
    variants = _available(cfg, rnd)
    scores = development_scores(cfg, variants)
    root = PROJECT_ROOT / rnd.out
    root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(root / "development.csv", index=False)
    mean = scores.pivot_table(
        index=["variant", "arch"], columns="period", values="skill_mae", aggfunc="mean"
    )
    ref = mean.loc[rnd.reference]
    best = mean["2008-2021"].idxmax()
    row = mean.loc[best]
    adopt = (
        best != rnd.reference
        and row["2008-2021"] > ref["2008-2021"]
        and row["2008-2013"] >= ref["2008-2013"]
    )
    hashes = {}
    for results_dir in variants.values():
        vp = get_paths({"paths": {"results_dir": results_dir}})
        for a in cfg["architectures"]:
            for h in cfg["targets"]["horizons"]:
                f = vp.predictions_csv(a, h)
                hashes[str(f.relative_to(PROJECT_ROOT))] = _sha(f)
    for doc in rnd.fixed:
        hashes[doc] = _sha(PROJECT_ROOT / doc)
    decision = {
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": rnd.rule,
        "best": {"variant": best[0], "arch": best[1], **row.to_dict()},
        "reference": {"variant": rnd.reference[0], "arch": rnd.reference[1], **ref.to_dict()},
        "adopt": bool(adopt),
        "chosen": {"variant": best[0], "arch": best[1]}
        if adopt
        else {"variant": rnd.reference[0], "arch": rnd.reference[1]},
        "scores": mean.reset_index().to_dict(orient="records"),
        "sha256": hashes,
    }
    write_json(root / "decision.json", decision)
    return decision


def test_report(cfg: Config, paths: Paths, rnd: Round = RONDA3) -> pd.DataFrame:
    """Test metrics of every variant (computed once, after ``decide``)."""
    root = PROJECT_ROOT / rnd.out
    decision = read_json(root / "decision.json")
    changed = [f for f, s in decision["sha256"].items() if _sha(PROJECT_ROOT / f) != s]
    if changed:
        raise RuntimeError(f"files changed after the decision: {changed}")
    variants = _available(cfg, rnd)
    panel = Panel.load(paths, cfg)
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    rows, years = [], []
    for h in cfg["targets"]["horizons"]:
        naive = naive_predictions(cfg, panel, h)
        naive_test = naive[naive["phase"] == "test"]
        for name, results_dir in variants.items():
            vp = get_paths({"paths": {"results_dir": results_dir}})
            for arch in cfg["architectures"]:
                pred = pd.read_csv(
                    vp.predictions_csv(arch, h), parse_dates=["origin", "target_date", "refit_date"]
                )
                ens = add_conformal(recalibrate_center(ensemble(pred, qcols), cfg), cfg)
                test = ens[ens["phase"] == "test"]
                m = frame_metrics(test, naive_test, h, cfg, calibrated=True)
                t = test.set_index("origin").sort_index()
                actual = t["price"] * np.exp(t["y"])
                dm_abs = diebold_mariano(
                    (t["price"] * np.exp(t["q50"]) - actual).to_numpy(),
                    (t["price"] - actual).to_numpy(),
                    h,
                    "absolute",
                )
                rows.append(
                    {
                        "variant": name,
                        "arch": arch,
                        "horizon": h,
                        "mae": m["mae"],
                        "rmse": m["rmse"],
                        "r2": m["r2"],
                        "skill_mae": m["skill_mae"],
                        "skill_rmse": m["skill_rmse"],
                        "dm_pvalue_sq": m["dm_pvalue"],
                        "dm_pvalue_abs": dm_abs["dm_pvalue"],
                        "dir_acc": m["dir_acc"],
                        "coverage80": m["coverage80"],
                        "width80": m["width80"],
                        "winkler80": m["winkler80"],
                        "mean_lam": float(test["lam"].mean()),
                        "n": m["n_test"],
                    }
                )
                err_m = (t["price"] * np.exp(t["q50"]) - actual).abs()
                err_n = (t["price"] - actual).abs()
                yb = pd.DataFrame({"m": err_m, "n": err_n}).groupby(t.index.year).mean()
                for year, r in yb.iterrows():
                    years.append(
                        {
                            "variant": name,
                            "arch": arch,
                            "horizon": h,
                            "year": year,
                            "skill_mae": 1 - r["m"] / r["n"],
                        }
                    )
    test = pd.DataFrame(rows)
    test.to_csv(root / "test.csv", index=False)
    pd.DataFrame(years).to_csv(root / "annual.csv", index=False)
    return test
