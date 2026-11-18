"""Stationary input features computed from the Brent-calendar dataset.

Every feature at date t uses only information available at the close of t (rolling windows
look backwards), so features never depend on the future. Prices enter as log returns,
spreads or deviations from their own moving averages, never as raw levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilcast.config import Config

RETURN_SERIES = {
    "r_brent": "brent",
    "r_wti": "wti",
    "r_gas": "gas",
    "r_gold": "gold",
    "r_copper": "copper",
    "r_dxy": "dxy",
    "r_eurusd": "usd_per_eur",
    "r_cny": "cny_per_usd",
}

# Human-readable description (Spanish) of the features produced with the default config.
FEATURE_DESCRIPTIONS = {
    "r_brent": "Rentabilidad logarítmica diaria del Brent",
    "r_wti": "Rentabilidad logarítmica diaria del WTI spot",
    "r_gas": "Rentabilidad logarítmica diaria del gas natural Henry Hub",
    "r_gold": "Rentabilidad logarítmica diaria del oro",
    "r_copper": "Rentabilidad logarítmica diaria del cobre",
    "r_dxy": "Rentabilidad logarítmica diaria del índice dólar",
    "r_eurusd": "Rentabilidad logarítmica diaria del USD/EUR",
    "r_cny": "Rentabilidad logarítmica diaria del CNY/USD",
    "log_vix": "Logaritmo del VIX (nivel de aversión al riesgo)",
    "ecb_rate": "Tipo de la facilidad de depósito del BCE (nivel, %)",
    "d_ecb_rate": "Variación del tipo del BCE en el último mes",
    "spread_brent_wti": "Diferencial log(Brent/WTI)",
    "wti_basis": "Base log(WTI spot / futuro WTI más próximo)",
    "wti_fut_range": "Rango intradía log(máximo/mínimo) del futuro WTI",
    "wti_fut_volume": "Volumen del futuro WTI relativo a su media de 3 meses (log)",
    "rv_21": "Volatilidad realizada del Brent (1 mes)",
    "rv_63": "Volatilidad realizada del Brent (3 meses)",
    "dev_ma63": "Desviación log del Brent respecto a su media de 63 días",
    "dev_ma252": "Desviación log del Brent respecto a su media de 252 días",
    "mom_252": "Momentum: variación log del Brent en 252 días",
}

# Extra features of the extended experiment (only when listed in ``features.extra``).
EXTRA_DESCRIPTIONS = {
    "gpr": "Riesgo geopolítico (índice Caldara-Iacoviello, media 7 días, log)",
    "gpr_shock": "Riesgo geopolítico frente a su media del último año",
    "gpr_act": "Actos geopolíticos (guerras, ataques), media 7 días, log",
    "epu": "Incertidumbre de política económica EE. UU., media 30 días, log",
    "spx_21": "Rentabilidad log del S&P 500 en el último mes",
    "infl_exp_21": "Cambio en un mes de la inflación esperada a 10 años (EE. UU.)",
    "ovx_vix": "Miedo propio del crudo: log(OVX / VIX)",
}
# Extras whose history starts after the modelling sample (OVX 2007, T10YIE 2003): before
# that they are 0 after scaling and an availability channel ``<name>_disp`` tells the network.
OPTIONAL_EXTRA = {"infl_exp_21", "ovx_vix"}


def compute_extra_features(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Extra features from the columns of ``data/processed/extra_dataset.csv`` (all lagged
    by their publication delay in :mod:`oilcast.data.extra`)."""
    defs = {
        "gpr": lambda: df["gpr_ma7"],
        "gpr_shock": lambda: df["gpr_ma7"] - df["gpr_ma7"].rolling(252, min_periods=252).mean(),
        "gpr_act": lambda: df["gpr_act_ma7"],
        "epu": lambda: df["epu_ma30"],
        "spx_21": lambda: _log(df["spx"]).diff(21),
        "infl_exp_21": lambda: df["infl_exp"].diff(21),
        "ovx_vix": lambda: _log(df["ovx"]) - _log(df["vix"]),
    }
    unknown = set(names) - defs.keys()
    if unknown:
        raise ValueError(f"unknown extra features: {sorted(unknown)}")
    out = pd.DataFrame(index=df.index)
    for name in names:
        out[name] = defs[name]()
        if name in OPTIONAL_EXTRA:
            out[f"{name}_disp"] = out[name].notna().astype(float)
    return out


def _log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def compute_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a (dates x FEATURES) frame; NaN where history is insufficient."""
    fcfg = cfg["features"]
    out = pd.DataFrame(index=df.index)
    for name, col in RETURN_SERIES.items():
        out[name] = _log(df[col]).diff()

    out["log_vix"] = _log(df["vix"])
    out["ecb_rate"] = df["ecb_rate"]
    out["d_ecb_rate"] = df["ecb_rate"].diff(fcfg["rate_change_window"])
    out["spread_brent_wti"] = _log(df["brent"]) - _log(df["wti"])
    out["wti_basis"] = _log(df["wti"]) - _log(df["wti_fut_close"])
    out["wti_fut_range"] = (_log(df["wti_fut_high"]) - _log(df["wti_fut_low"])).clip(lower=0)
    log_vol = _log(df["wti_fut_volume"])
    vw = fcfg["volume_window"]
    out["wti_fut_volume"] = log_vol - log_vol.rolling(vw, min_periods=vw).mean()

    for w in fcfg["vol_windows"]:
        out[f"rv_{w}"] = out["r_brent"].rolling(w, min_periods=w).std()
    log_p = _log(df["brent"])
    for w in fcfg["ma_windows"]:
        out[f"dev_ma{w}"] = log_p - np.log(df["brent"].rolling(w, min_periods=w).mean())
    mw = fcfg["momentum_window"]
    out[f"mom_{mw}"] = log_p - log_p.shift(mw)
    if fcfg.get("extra"):
        out = out.join(compute_extra_features(df, fcfg["extra"]))
    for name in fcfg.get("optional", []):  # late-starting base series (long history)
        out[f"{name}_disp"] = out[name].notna().astype(float)
    return out
