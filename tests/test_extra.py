"""Extra features of the extended experiment: publication lags, optional history, no leakage."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from oilcast.data.extra import _lagged
from oilcast.features.windows import Panel


def _with_extra(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = len(df)
    out = df.copy()
    out["gpr_ma7"] = np.log1p(100 * np.exp(rng.normal(0, 0.3, n)))
    out["gpr_act_ma7"] = np.log1p(100 * np.exp(rng.normal(0, 0.3, n)))
    out["epu_ma30"] = np.log1p(100 * np.exp(rng.normal(0, 0.2, n)))
    out["spx"] = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ovx = 30 * np.exp(rng.normal(0, 0.2, n))
    ovx[out.index < "2007-05-10"] = np.nan  # real OVX starts in 2007
    out["ovx"] = ovx
    out["infl_exp"] = 2 + np.cumsum(rng.normal(0, 0.01, n))
    return out


def _cfg_with(cfg: dict, extra: list[str]) -> dict:
    c = copy.deepcopy(cfg)
    c["features"]["extra"] = extra
    return c


def test_lag_delays_availability() -> None:
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    lagged = _lagged(s, 8)
    assert list(lagged.index) == list(pd.to_datetime(["2020-01-09", "2020-01-10"]))


def test_optional_extra_keeps_the_same_origins(df, cfg, panel) -> None:
    c = _cfg_with(cfg, ["spx_21", "infl_exp_21", "ovx_vix", "gpr", "gpr_shock", "gpr_act", "epu"])
    p = Panel.from_dataset(_with_extra(df), c)
    assert p.optional == {"infl_exp_21", "ovx_vix"}
    assert "ovx_vix_disp" in p.feature_names
    for h in (5, 63, 252):
        np.testing.assert_array_equal(p.usable(h), panel.usable(h))


def test_extra_features_do_not_look_ahead(df, cfg) -> None:
    c = _cfg_with(cfg, ["spx_21", "gpr_shock", "ovx_vix"])
    full = _with_extra(df)
    cut = 5000
    changed = full.copy()
    changed.iloc[cut + 1 :, :] *= 1.7  # alter everything after the origin
    a = Panel.from_dataset(full, c).features[: cut + 1]
    b = Panel.from_dataset(changed, c).features[: cut + 1]
    np.testing.assert_allclose(a, b, equal_nan=True)
