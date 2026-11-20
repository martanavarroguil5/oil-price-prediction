"""Ronda 5: candidatas por plazo para mejorar la propuesta B (docs/experimento-ronda5.md).

``decide5`` puntúa todas las candidatas solo con desarrollo, aplica la regla y guarda la
decisión con los hashes de las predicciones. ``test5`` se niega a ejecutarse sin ella y
calcula el test una sola vez, con intervalos bootstrap por bloques.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oilcast.config import Config, load_config
from oilcast.evaluation.metrics import diebold_mariano
from oilcast.evaluation.summary import ensemble, recalibrate_center
from oilcast.paths import PROJECT_ROOT, get_paths
from oilcast.training.evaluate import quantile_columns
from oilcast.utils import read_json, write_json

OUT = Path("reports/experiments/ronda5")
W1 = "reports/experiments/ronda3/w1"
ORO = "reports/experiments/ronda4/oro"
W1_20 = "reports/experiments/ronda5/w1_20"
ORO_20 = "reports/experiments/ronda5/oro_20"
FUT = "reports/experiments/ronda5/futuros"
ARCHS = ("rnn", "lstm", "gru")
MIN_GAIN = 0.005  # 0,5 puntos de mejora del MAE en 2008-2021 frente al componente de B
CANDIDATES: dict[int, dict[str, list[tuple[str, str]]]] = {
    5: {
        "B: RNN W1": [(W1, "rnn")],
        "W1, tres redes": [(W1, a) for a in ARCHS],
        "RNN W1, 20 semillas": [(W1_20, "rnn")],
        "W1, tres redes, 20 semillas": [(W1_20, a) for a in ARCHS],
        "RNN con futuros": [(FUT, "rnn")],
        "LSTM con futuros": [(FUT, "lstm")],
        "GRU con futuros": [(FUT, "gru")],
        "Con futuros, tres redes": [(FUT, a) for a in ARCHS],
    },
    63: {
        "B: RNN W1": [(W1, "rnn")],
        "W1, tres redes": [(W1, a) for a in ARCHS],
        "RNN W1, 20 semillas": [(W1_20, "rnn")],
        "W1, tres redes, 20 semillas": [(W1_20, a) for a in ARCHS],
    },
    252: {
        "B: RNN oro/crudo": [(ORO, "rnn")],
        "Oro/crudo, tres redes": [(ORO, a) for a in ARCHS],
        "RNN oro/crudo, 20 semillas": [(ORO_20, "rnn")],
        "Oro/crudo, tres redes, 20 semillas": [(ORO_20, a) for a in ARCHS],
    },
}
FIXED = (
    "docs/experimento-ronda5.md",
    "configs/ronda5_futuros.yaml",
    "configs/ronda5_semillas_w1.yaml",
    "configs/ronda5_semillas_oro.yaml",
)
PERIODS = {
    "2008-2013": ("2008-01-01", "2014-01-01"),
    "2014-2021": ("2014-01-01", "2022-01-01"),
    "2008-2021": ("2008-01-01", "2022-01-01"),
}


def _cfg() -> Config:
    return load_config(PROJECT_ROOT / "configs/ronda3_w1.yaml")


def _file(results_dir: str, arch: str, h: int) -> Path:
    return get_paths({"paths": {"results_dir": results_dir}}).predictions_csv(arch, h)


def adoptable(members: list[tuple[str, str]]) -> bool:
    """Solo una RNN: la aplicación integra un único tipo de red (enunciado)."""
    return len(members) == 1 and members[0][1] == "rnn"


def _members(members: list[tuple[str, str]], h: int) -> pd.DataFrame:
    frames = []
    for i, (results_dir, arch) in enumerate(members):
        p = pd.read_csv(
            _file(results_dir, arch, h), parse_dates=["origin", "target_date", "refit_date"]
        )
        p["seed"] = 1000 * i + p["seed"]  # la media de todas las redes y semillas
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def _skill(d: pd.DataFrame) -> float:
    actual = d["price"] * np.exp(d["y"])
    model = d["price"] * np.exp(d["q50"])
    return float(1 - (model - actual).abs().mean() / (d["price"] - actual).abs().mean())


def _rmse_skill(d: pd.DataFrame) -> float:
    actual = d["price"] * np.exp(d["y"])
    e_m = d["price"] * np.exp(d["q50"]) - actual
    e_n = d["price"] - actual
    return float(1 - np.sqrt((e_m**2).mean()) / np.sqrt((e_n**2).mean()))


def _bootstrap(d: pd.DataFrame, h: int, n: int = 2000, seed: int = 5) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True)
    m, block = len(d), max(h, 21)
    vals = []
    for _ in range(n):
        starts = rng.integers(0, m, size=int(np.ceil(m / block)))
        idx = np.concatenate([(s + np.arange(block)) % m for s in starts])[:m]
        vals.append(_skill(d.iloc[idx]))
    lo, hi = np.percentile(vals, [5, 95])
    return float(lo), float(hi)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development(cfg: Config) -> pd.DataFrame:
    """Mejora del MAE en desarrollo (solo reentrenos de calibración, objetivo anterior a 2022)."""
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    test_start = np.datetime64(pd.Timestamp(cfg["splits"]["test_start"]))
    rows = []
    for h, cands in CANDIDATES.items():
        for name, members in cands.items():
            pred = _members(members, h)
            pred = pred[pred["phase"] == "calibration"]
            ens = recalibrate_center(ensemble(pred, qcols), cfg)
            ens = ens[ens["target_date"].values < test_start]
            for period, (a, b) in PERIODS.items():
                s = ens[(ens["origin"] >= a) & (ens["origin"] < b)]
                rows.append(
                    {
                        "horizon": h,
                        "candidate": name,
                        "adoptable": adoptable(members),
                        "period": period,
                        "skill_mae": _skill(s),
                        "n": len(s),
                    }
                )
    return pd.DataFrame(rows)


def decide5(cfg: Config | None = None) -> dict:
    cfg = cfg or _cfg()
    root = PROJECT_ROOT / OUT
    root.mkdir(parents=True, exist_ok=True)
    dev = development(cfg)
    dev.to_csv(root / "development.csv", index=False)
    table = dev.pivot_table(index=["horizon", "candidate"], columns="period", values="skill_mae")
    choice = {}
    for h, cands in CANDIDATES.items():
        ref_name = next(iter(cands))
        ref = table.loc[(h, ref_name)]
        worth = []
        for name, members in cands.items():
            if name == ref_name:
                continue
            row = table.loc[(h, name)]
            ok = (
                row["2008-2021"] >= ref["2008-2021"] + MIN_GAIN
                and row["2008-2013"] >= ref["2008-2013"]
                and row["2014-2021"] >= ref["2014-2021"]
            )
            if ok:
                worth.append((row["2008-2021"], name, adoptable(members)))
        rnn = [w for w in worth if w[2]]
        choice[str(h)] = {
            "reference": ref_name,
            "worth_it": [w[1] for w in sorted(worth, reverse=True)],
            "chosen": max(rnn)[1] if rnn else ref_name,
        }
    hashes = {}
    for h, cands in CANDIDATES.items():
        for members in cands.values():
            for results_dir, arch in members:
                f = _file(results_dir, arch, h)
                hashes[str(f.relative_to(PROJECT_ROOT))] = _sha(f)
    for doc in FIXED:
        hashes[doc] = _sha(PROJECT_ROOT / doc)
    decision = {
        "decided_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": "docs/experimento-ronda5.md: mejora >= 0,5 puntos en 2008-2021 frente a B, sin "
        "empeorar en 2008-2013 ni en 2014-2021; solo se adoptan candidatas RNN.",
        "choice": choice,
        "development": table.reset_index().to_dict(orient="records"),
        "sha256": hashes,
    }
    write_json(root / "decision.json", decision)
    return decision


def test5(cfg: Config | None = None) -> pd.DataFrame:
    """Test 2022-2026 de todas las candidatas (una sola vez, después de ``decide5``)."""
    cfg = cfg or _cfg()
    root = PROJECT_ROOT / OUT
    decision = read_json(root / "decision.json")
    changed = [f for f, s in decision["sha256"].items() if _sha(PROJECT_ROOT / f) != s]
    if changed:
        raise RuntimeError(f"files changed after the decision: {changed}")
    qcols = quantile_columns(cfg["targets"]["quantiles"])
    rows, years = [], []
    for h, cands in CANDIDATES.items():
        for name, members in cands.items():
            ens = recalibrate_center(ensemble(_members(members, h), qcols), cfg)
            ens = ens[ens["y"].notna()]
            test = ens[ens["phase"] == "test"].sort_values("origin")
            oos = ens[ens["origin"] >= "2008-01-01"].sort_values("origin")
            actual = test["price"] * np.exp(test["y"])
            dm = diebold_mariano(
                (test["price"] * np.exp(test["q50"]) - actual).to_numpy(),
                (test["price"] - actual).to_numpy(),
                h,
                "absolute",
            )
            lo, hi = _bootstrap(test, h)
            olo, ohi = _bootstrap(oos, h)
            rows.append(
                {
                    "horizon": h,
                    "candidate": name,
                    "adoptable": adoptable(members),
                    "chosen": decision["choice"][str(h)]["chosen"] == name,
                    "skill_mae": _skill(test),
                    "skill_rmse": _rmse_skill(test),
                    "dm_pvalue_abs": dm["dm_pvalue"],
                    "boot90_lo": lo,
                    "boot90_hi": hi,
                    "oos_skill_mae": _skill(oos),
                    "oos_boot90_lo": olo,
                    "oos_boot90_hi": ohi,
                    "n": len(test),
                }
            )
            for year, g in test.groupby(test["origin"].dt.year):
                years.append(
                    {"horizon": h, "candidate": name, "year": year, "skill_mae": _skill(g)}
                )
    out = pd.DataFrame(rows)
    out.to_csv(root / "test.csv", index=False)
    pd.DataFrame(years).to_csv(root / "annual.csv", index=False)
    return out
