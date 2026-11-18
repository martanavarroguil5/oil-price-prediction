"""Extra daily series for the extended experiment (geopolitics, uncertainty, macro).

They are kept apart from the main dataset so the baseline results stay reproducible:
raw files go to ``data/raw/extra`` and the aligned table to
``data/processed/extra_dataset.csv``, on exactly the same Brent calendar as the main dataset.

Publication lags are respected: a value dated ``d`` is only used from ``d + lag`` onwards,
so the backtest never sees an index before it was actually published.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from oilcast.config import Config
from oilcast.data.dataset import align_to_calendar, load_dataset
from oilcast.data.download import _save, _with_retries, fetch_fred, fetch_yahoo
from oilcast.paths import Paths

log = logging.getLogger(__name__)

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

# name -> publication lag in calendar days (value dated d is usable from d + lag)
LAGS = {"gpr": 8, "gpr_act": 8, "epu": 1, "ovx": 0, "infl_exp": 0, "spx": 0}


def fetch_gpr(timeout: float, retries: int) -> pd.DataFrame:
    """Caldara-Iacoviello daily geopolitical risk index (total and 'acts')."""

    def call() -> pd.DataFrame:
        resp = requests.get(GPR_URL, timeout=timeout)
        resp.raise_for_status()
        raw = pd.read_excel(io.BytesIO(resp.content), engine="xlrd")
        out = raw.set_index(pd.to_datetime(raw["date"]))[["GPRD", "GPRD_ACT"]]
        out.columns = ["gpr", "gpr_act"]
        return out.dropna(how="all")

    return _with_retries(call, retries, "GPR daily")


def download_extra(cfg: Config, paths: Paths) -> dict[str, str]:
    dcfg = cfg["data"]
    start, end = dcfg["start_date"], (date.today() + timedelta(days=1)).isoformat()
    timeout, retries = dcfg["timeout_s"], dcfg["retries"]
    out_dir = paths.raw / "extra"
    jobs = {
        "gpr": lambda: fetch_gpr(timeout, retries),
        "epu": lambda: fetch_fred("USEPUINDXD", start, end, timeout, retries),
        "ovx": lambda: fetch_fred("OVXCLS", start, end, timeout, retries),
        "infl_exp": lambda: fetch_fred("T10YIE", start, end, timeout, retries),
        "spx": lambda: fetch_yahoo("^GSPC", start, end, retries)["close"],
    }
    status = {}
    for name, fn in jobs.items():
        try:
            obj = fn()
            _save(obj, out_dir / f"{name}.csv", name if isinstance(obj, pd.Series) else None)
            status[name] = f"ok ({len(obj)} filas, última {obj.index.max().date()})"
        except Exception as exc:  # noqa: BLE001
            status[name] = f"error: {exc}"
            log.warning("download extra %s failed: %s", name, exc)
    for name, msg in status.items():
        log.info("%-10s %s", name, msg)
    return status


def _raw(paths: Paths, name: str) -> pd.DataFrame:
    path = paths.raw / "extra" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `oilcast download-extra`")
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


def _lagged(s: pd.Series, lag_days: int) -> pd.Series:
    s = s.dropna()
    return pd.Series(s.to_numpy(), index=s.index + pd.Timedelta(days=lag_days))


def build_extra(paths: Paths) -> pd.DataFrame:
    """Align the extra series to the main dataset's calendar (as of each Brent date)."""
    calendar = load_dataset(paths).index
    gpr = _raw(paths, "gpr")
    daily = {
        # 7-calendar-day mean so weekend news counts too; log1p because the index can be 0
        "gpr_ma7": np.log1p(gpr["gpr"]).rolling("7D").mean(),
        "gpr_act_ma7": np.log1p(gpr["gpr_act"]).rolling("7D").mean(),
        "epu_ma30": np.log1p(_raw(paths, "epu")["epu"]).rolling("30D").mean(),
    }
    lags = {"gpr_ma7": LAGS["gpr"], "gpr_act_ma7": LAGS["gpr_act"], "epu_ma30": LAGS["epu"]}
    df = pd.DataFrame(index=calendar)
    for col, s in daily.items():
        df[col] = align_to_calendar(_lagged(s, lags[col]), calendar)
    for name in ("ovx", "infl_exp", "spx"):
        df[name] = align_to_calendar(_lagged(_raw(paths, name)[name], LAGS[name]), calendar)
    df.index.name = "date"
    df.to_csv(paths.extra_csv, float_format="%.6g")
    log.info("extra dataset: %d rows, columns %s", len(df), list(df.columns))
    return df


def load_extra(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.extra_csv, index_col="date", parse_dates=["date"]).astype(float)
