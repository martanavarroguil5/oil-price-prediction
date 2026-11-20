"""Ronda 4, criba de 20 predictores con regresión lineal recursiva (solo objetivo < 2022).

Uso: python scripts/ronda4_criba.py .   (la ruta es la raíz del proyecto)
"""

import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1]
d = pd.read_csv(f"{ROOT}/data/processed/brent_dataset.csv", parse_dates=["date"], index_col="date")
for name in ("gasoline", "heating_oil"):
    s = pd.read_csv(f"{ROOT}/data/raw/extra/{name}.csv", parse_dates=["date"], index_col="date")
    d[name] = s[name].reindex(d.index).ffill()
lp = np.log(d["brent"])
idx = d.index
n = len(d)


def lg(s):
    return np.log(s.where(s > 0)).ffill()


def dev(s, w):
    return s - s.rolling(w, min_periods=w // 2).mean()


X = pd.DataFrame(index=idx)
X["dev_ma252"] = lp - np.log(d["brent"].rolling(252, min_periods=252).mean())
X["dev_ma756"] = lp - np.log(d["brent"].rolling(756, min_periods=500).mean())
X["mom_252"] = lp - lp.shift(252)
X["mom_63"] = lp - lp.shift(63)
X["mom_21"] = lp - lp.shift(21)
X["cu_252"] = lg(d["copper"]).diff(252)
X["cu_63"] = lg(d["copper"]).diff(63)
X["cu_21"] = lg(d["copper"]).diff(21)
X["cu_oil"] = dev(lg(d["copper"]) - lp, 756)
X["au_oil"] = dev(lg(d["gold"]) - lp, 756)
X["gas_oil"] = dev(lg(d["gas"]) - lp, 756)
X["basis_wti"] = lg(d["wti"]) - lg(d["wti_fut_close"])
X["basis_brent"] = (lg(d["brent"]) - lg(d["brent_fut_close"])).fillna(0.0)
X["crack_gas"] = dev(lg(d["gasoline"] * 42) - lp, 252)
X["crack_ho"] = dev(lg(d["heating_oil"] * 42) - lp, 252)
X["dxy_63"] = lg(d["dxy"]).diff(63)
X["dxy_252"] = lg(d["dxy"]).diff(252)
X["log_vix"] = lg(d["vix"])
X["rv_63"] = lp.diff().rolling(63).std()
X["brent_wti"] = lg(d["brent"]) - lg(d["wti"])
X = X.fillna(0.0)

REFITS = [pd.Timestamp(f"{y}-01-01") for y in range(2008, 2022)]
PERIODS = {"2008-2013": ("2008-01-01", "2014-01-01"), "2014-2021": ("2014-01-01", "2022-01-01")}


def run(h):
    y = (lp.shift(-h) - lp).to_numpy()
    tdate = pd.Series(idx).shift(-h).to_numpy()
    preds = {c: np.full(n, np.nan) for c in X.columns}
    preds["ridge"] = np.full(n, np.nan)
    for i, r in enumerate(REFITS):
        nxt = REFITS[i + 1] if i + 1 < len(REFITS) else pd.Timestamp("2022-01-01")
        tr = np.where((idx >= "2000-09-01") & (tdate < np.datetime64(r)) & np.isfinite(y))[0]
        pr = np.where((idx >= r) & (idx < nxt))[0]
        yt = y[tr]
        Z = X.to_numpy()
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-12
        Zs = (Z - mu) / sd
        for j, c in enumerate(X.columns):
            x = Zs[tr, j]
            b = (x @ yt) / (x @ x)
            preds[c][pr] = b * Zs[pr, j]
        A = Zs[tr]
        lam = 10.0 * len(tr)
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yt)
        preds["ridge"][pr] = Zs[pr] @ beta
    preds["combo"] = np.nanmean(np.vstack([preds[c] for c in X.columns]), axis=0)
    price = d["brent"].to_numpy()
    actual = price * np.exp(y)
    rows = []
    for name, p in preds.items():
        row = {"h": h, "model": name}
        for per, (a, b) in PERIODS.items():
            m = (idx >= a) & (idx < b) & (tdate < np.datetime64("2022-01-01")) & np.isfinite(p)
            m &= np.isfinite(y)
            e_m = np.abs(price[m] * np.exp(p[m]) - actual[m]).mean()
            e_n = np.abs(price[m] - actual[m]).mean()
            row[per] = 100 * (1 - e_m / e_n)
        rows.append(row)
    return pd.DataFrame(rows)


out = pd.concat([run(h) for h in (5, 63, 252)])
out["min"] = out[["2008-2013", "2014-2021"]].min(axis=1)
pd.set_option("display.width", 200)
print(out.round(1).to_string(index=False))
