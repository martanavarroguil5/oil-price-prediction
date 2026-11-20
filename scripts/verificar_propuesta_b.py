"""Comprobaciones de la propuesta B (RNN W1 a 5 y 63 sesiones, RNN R4 a 252).

Uso: python scripts/verificar_propuesta_b.py
Escribe reports/experiments/propuesta_b/verificacion.csv y placebo.csv e imprime un resumen.

1. Ninguna variable mira al futuro: las variables calculadas con datos cortados en una fecha
   coinciden con las calculadas con toda la serie.
2. Cada reentreno solo usa objetivos realizados antes de su fecha y predice orígenes
   posteriores; la búsqueda de hiperparámetros solo validó en 2014-2021.
3. Recalculo independiente del test (ensemble, recalibración del centro y MAE frente al
   ingenuo) sin las funciones de evaluación del proyecto.
4. Solidez: por año, sin los orígenes de 2025, bootstrap por bloques, por semilla, con el oro
   retrasado un día y frente a variables placebo sin información.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from oilcast.config import for_horizon, load_config
from oilcast.data.dataset import load_dataset
from oilcast.features.engineering import compute_features
from oilcast.features.windows import Panel
from oilcast.paths import PROJECT_ROOT, get_paths
from oilcast.training.splits import purged_train

OUT = PROJECT_ROOT / "reports/experiments/propuesta_b"
W1_CFG = load_config(PROJECT_ROOT / "configs/ronda3_w1.yaml")
R4_CFG = load_config(PROJECT_ROOT / "configs/ronda4.yaml")
PIECES = {  # horizonte -> (configuración, predicciones, búsqueda)
    5: (W1_CFG, "reports/experiments/ronda3/w1/predictions/rnn_h5.csv", "reports/results"),
    63: (W1_CFG, "reports/experiments/ronda3/w1/predictions/rnn_h63.csv", "reports/results"),
    252: (
        R4_CFG,
        "reports/experiments/ronda4/oro/predictions/rnn_h252.csv",
        "reports/experiments/ronda4/oro",
    ),
}
QCOLS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
TEST_START = np.datetime64("2022-01-01")
rng = np.random.default_rng(42)
rows: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    rows.append({"comprobacion": name, "ok": bool(ok), "detalle": detail})
    print(("OK   " if ok else "FALLO") + f" {name} {detail}")


# 1. Sin mirar al futuro -------------------------------------------------------------------
df = load_dataset(get_paths(W1_CFG))
for label, cfg in (("20 variables W1", W1_CFG), ("oro_crudo", R4_CFG)):
    full = compute_features(df, cfg)
    worst = 0.0
    for cut in ("2008-06-30", "2012-12-31", "2016-03-31", "2020-06-30", "2022-12-30", "2025-10-31"):
        part = compute_features(df.loc[:cut], cfg)
        a, b = part.to_numpy(), full.loc[: part.index[-1]].to_numpy()
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b))
        worst = max(worst, float((~same).mean()))
    check(f"sin datos futuros: {label}", worst == 0.0, f"fracción distinta {worst:.2e}")


# 2. Purga, calendario de reentrenos y búsqueda ---------------------------------------------
def load_pred(path: str) -> pd.DataFrame:
    return pd.read_csv(
        PROJECT_ROOT / path, parse_dates=["origin", "target_date", "refit_date"]
    ).sort_values(["origin", "seed"])


preds = {h: load_pred(p) for h, (_, p, _) in PIECES.items()}
panels = {h: Panel.load(get_paths(c), for_horizon(c, h)) for h, (c, _, _) in PIECES.items()}
for h, pred in preds.items():
    panel = panels[h]
    bad_order = int((pred["origin"] < pred["refit_date"]).sum())
    leaks = 0
    for refit in pred["refit_date"].unique():
        train = purged_train(panel, h, pd.Timestamp(refit))
        leaks += int((panel.target_dates(h)[train] >= np.datetime64(refit)).sum())
    test_refits = sorted(pd.to_datetime(pred.loc[pred.phase == "test", "refit_date"].unique()))
    yearly = all(d.month == 1 and d.day == 1 for d in test_refits)
    check(
        f"h={h}: reentrenos sin objetivos posteriores",
        bad_order == 0 and leaks == 0 and yearly,
        f"orígenes antes del reentreno {bad_order}, objetivos filtrados {leaks}, "
        f"reentrenos de test {[d.year for d in test_refits]}",
    )
for h, (_, _, tdir) in PIECES.items():
    if tdir is None:
        continue
    best = json.loads((PROJECT_ROOT / tdir / f"tuning/rnn_h{h}/best.json").read_text())
    folds = best.get("fold_names", [])
    check(
        f"h={h}: búsqueda validada solo en 2014-2021",
        all(f[:4] >= "2014" and f[-4:] <= "2022" for f in folds) and len(folds) == 4,
        f"bloques {folds}",
    )


# 3. Recalculo independiente ----------------------------------------------------------------
def shrink(origin, tdate, pred, y, min_known=100):
    """Factor del centro con los resultados ya conocidos antes de cada origen."""
    lam = np.zeros(len(origin))
    for i, o in enumerate(origin):
        known = tdate < o
        if known.sum() >= min_known:
            p, t = pred[known], y[known]
            lam[i] = np.clip((p @ t) / (p @ p), 0.0, 1.0) if p @ p > 0 else 0.0
    return lam


def central(pred: pd.DataFrame) -> pd.DataFrame:
    g = pred.groupby("origin")
    ens = g[QCOLS].mean()
    ens[:] = np.sort(ens.to_numpy(), axis=1)
    first = g[["target_date", "price", "y", "phase"]].first()
    out = first.join(ens[["q50"]]).reset_index()
    out["lam"] = shrink(
        out["origin"].to_numpy(),
        out["target_date"].to_numpy(),
        out["q50"].to_numpy(),
        out["y"].to_numpy(),
    )
    return out


def skill(d: pd.DataFrame) -> float:
    actual = d["price"] * np.exp(d["y"])
    model = d["price"] * np.exp(d["lam"] * d["q50"])
    return 1 - (model - actual).abs().mean() / (d["price"] - actual).abs().mean()


reported = {5: 0.00620, 63: 0.03727, 252: 0.09083}
tests = {}
for h, pred in preds.items():
    c = central(pred)
    t = c[c["phase"] == "test"].copy()
    tests[h] = t
    s = skill(t)
    check(
        f"h={h}: recalculo independiente del test",
        abs(s - reported[h]) < 5e-4,
        f"{100 * s:+.2f} % (publicado {100 * reported[h]:+.2f} %)",
    )
    rows.append({"comprobacion": f"h={h}: mejora del MAE en test", "valor": s})

# 4. Solidez --------------------------------------------------------------------------------
for h, t in tests.items():
    for year, g in t.groupby(t["origin"].dt.year):
        rows.append({"comprobacion": f"h={h}: año {year}", "valor": skill(g), "n": len(g)})
    # bootstrap circular por bloques de h sesiones (las previsiones se solapan)
    n, block = len(t), max(h, 21)
    vals = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        vals.append(skill(t.iloc[idx]))
    lo, hi = np.percentile(vals, [5, 95])
    share = float(np.mean(np.array(vals) > 0))
    rows.append({"comprobacion": f"h={h}: bootstrap 90 %", "valor": lo, "valor2": hi, "n": share})
    print(f"     h={h}: bootstrap 90 % [{100 * lo:+.1f} %, {100 * hi:+.1f} %], P(>0)={share:.2f}")
    # semillas por separado
    seeds = []
    for _, g in preds[h].groupby("seed"):
        cs = central(g)
        seeds.append(skill(cs[cs["phase"] == "test"]))
    rows.append(
        {"comprobacion": f"h={h}: semillas (mín-máx)", "valor": min(seeds), "valor2": max(seeds)}
    )
    print(f"     h={h}: por semilla {', '.join(f'{100 * s:+.1f}' for s in seeds)}")

t252 = tests[252]
no25 = skill(t252[t252["origin"].dt.year < 2025])
rows.append({"comprobacion": "h=252: sin orígenes de 2025", "valor": no25})
print(f"     h=252 sin orígenes de 2025: {100 * no25:+.1f} %")


# Placebo: la misma regresión recursiva con variables sin información sobre el crudo
def linear_skill(x: np.ndarray, panel: Panel, h: int, cfg) -> float:
    y = panel.target(h)
    td = panel.target_dates(h)
    ok = panel.usable(h) & np.isfinite(x)
    dates = panel.dates.values
    refits = [np.datetime64(d) for d in cfg["splits"]["calibration_refits"]]
    refits += [np.datetime64(f"{yr}-01-01") for yr in range(2022, 2027)]
    frames = []
    for i, r in enumerate(refits):
        end = refits[i + 1] if i + 1 < len(refits) else dates[-1] + np.timedelta64(1, "D")
        if r < TEST_START:
            end = min(end, TEST_START)
        tr = np.flatnonzero(ok & (td < r))
        pr = np.flatnonzero(ok & (dates >= r) & (dates < end))
        xc = x[tr] - x[tr].mean()
        beta = (xc @ (y[tr] - y[tr].mean())) / (xc @ xc)
        frames.append(
            pd.DataFrame(
                {
                    "origin": dates[pr],
                    "target_date": td[pr],
                    "price": panel.price[pr],
                    "y": y[pr],
                    "q50": beta * (x[pr] - x[tr].mean()),
                    "phase": np.where(dates[pr] >= TEST_START, "test", "calibration"),
                }
            )
        )
    d = pd.concat(frames, ignore_index=True).dropna(subset=["y"])
    d["lam"] = shrink(
        d["origin"].to_numpy(),
        d["target_date"].to_numpy(),
        d["q50"].to_numpy(),
        d["y"].to_numpy(),
    )
    return skill(d[d["phase"] == "test"])


panel = panels[252]
x_real = panel.features[:, 0].astype(float)
real = linear_skill(x_real, panel, 252, R4_CFG)
gold = np.log(df["gold"].ffill().to_numpy())
brent = np.log(df["brent"].to_numpy())
lag = pd.Series(gold).shift(1).to_numpy() - brent
lag = lag - pd.Series(lag).rolling(756, min_periods=252).mean().to_numpy()
lagged = linear_skill(lag, panel, 252, R4_CFG)
print(f"     lineal oro/crudo {100 * real:+.1f} %, con el oro de la víspera {100 * lagged:+.1f} %")
rows.append({"comprobacion": "h=252: lineal oro/crudo", "valor": real})
rows.append({"comprobacion": "h=252: lineal con el oro de la víspera", "valor": lagged})

dgold = np.diff(gold[np.isfinite(gold)])
placebo = []
for kind in ("ruido", "oro falso"):
    for _ in range(300):
        steps = rng.normal(0, dgold.std(), size=len(gold))
        walk = np.cumsum(steps)
        raw = walk if kind == "ruido" else walk - brent
        x = raw - pd.Series(raw).rolling(756, min_periods=252).mean().to_numpy()
        placebo.append({"tipo": kind, "mejora": linear_skill(x, panel, 252, R4_CFG)})
pl = pd.DataFrame(placebo)
OUT.mkdir(parents=True, exist_ok=True)
pl.to_csv(OUT / "placebo.csv", index=False)
for kind, g in pl.groupby("tipo"):
    share = float((g["mejora"] >= real).mean())
    q = np.percentile(g["mejora"], [50, 90, 95])
    rows.append({"comprobacion": f"placebo {kind}: P(≥ lineal real)", "valor": share, "n": len(g)})
    print(
        f"     placebo {kind}: mediana {100 * q[0]:+.1f} %, p90 {100 * q[1]:+.1f} %, "
        f"p95 {100 * q[2]:+.1f} %, P(≥ real) = {share:.2f}"
    )
pd.DataFrame(rows).to_csv(OUT / "verificacion.csv", index=False)
