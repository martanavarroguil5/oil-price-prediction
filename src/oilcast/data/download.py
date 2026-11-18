"""Download raw daily series (FRED, Yahoo Finance, OPEC best-effort) into ``data/raw``.

Raw files are stored exactly as published (no filling, no calendar alignment): one CSV per
source with a ``date`` index. Alignment and cleaning happen in :mod:`oilcast.data.dataset`.
If a source fails (offline, rate limit...), the previously stored raw file is kept.
"""

from __future__ import annotations

import io
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
import requests

from oilcast.config import Config
from oilcast.paths import Paths

log = logging.getLogger(__name__)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
OHLCV = ["open", "high", "low", "close", "volume"]


def _with_retries[T](fn: Callable[[], T], retries: int, what: str) -> T:
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network errors of many kinds
            if attempt == retries:
                raise
            wait = 2**attempt
            log.warning("%s failed (%s); retry %d/%d in %ds", what, exc, attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_fred(series_id: str, start: str, end: str, timeout: float, retries: int) -> pd.Series:
    """Daily FRED series; missing observations ('.') stay NaN and are dropped."""

    def call() -> pd.Series:
        resp = requests.get(
            FRED_URL,
            params={"id": series_id, "cosd": start, "coed": end},
            timeout=timeout,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), na_values=".")
        s = pd.Series(df.iloc[:, 1].to_numpy(dtype=float), index=pd.to_datetime(df.iloc[:, 0]))
        return s.dropna()

    return _with_retries(call, retries, f"FRED {series_id}")


def fetch_yahoo(ticker: str, start: str, end: str, retries: int) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance (unadjusted). Raises if nothing is returned."""
    import yfinance as yf

    def call() -> pd.DataFrame:
        df = yf.download(
            ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False
        )
        if df is None or df.empty:
            raise RuntimeError(f"empty response for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[OHLCV]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df.dropna(how="all")

    return _with_retries(call, retries, f"Yahoo {ticker}")


def parse_opec_xml(content: bytes) -> pd.Series:
    """Parse the OPEC basket archive: elements carrying a date and a value attribute."""
    root = ET.fromstring(content)
    dates, values = [], []
    for el in root.iter():
        attrs = {k.lower(): v for k, v in el.attrib.items()}
        d = attrs.get("data") or attrs.get("date")
        v = attrs.get("val") or attrs.get("value")
        if d and v:
            dates.append(pd.to_datetime(d))
            values.append(float(v))
    if not dates:
        raise ValueError("no basket prices found in OPEC XML")
    return pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()


def fetch_opec(url: str, timeout: float) -> pd.Series:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return parse_opec_xml(resp.content)


def _save(obj: pd.Series | pd.DataFrame, path, name: str | None = None) -> None:
    df = obj.to_frame(name) if isinstance(obj, pd.Series) else obj
    df.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(path)


def download_all(cfg: Config, paths: Paths) -> dict[str, str]:
    """Download every source to ``data/raw``. Returns a status message per source."""
    dcfg = cfg["data"]
    start = dcfg["start_date"]
    end = (date.today() + timedelta(days=1)).isoformat()
    timeout, retries = dcfg["timeout_s"], dcfg["retries"]
    status: dict[str, str] = {}

    def attempt(name: str, fn: Callable[[], pd.Series | pd.DataFrame]) -> None:
        try:
            obj = fn()
            _save(obj, paths.raw / f"{name}.csv", name if isinstance(obj, pd.Series) else None)
            last = obj.index.max().date()
            status[name] = f"ok ({len(obj)} filas, última {last})"
        except Exception as exc:  # noqa: BLE001
            kept = (paths.raw / f"{name}.csv").exists()
            status[name] = f"error: {exc}" + (" (se conserva la copia local)" if kept else "")
            log.warning("download %s failed: %s", name, exc)

    for series_id, name in dcfg["fred"].items():
        attempt(name, lambda s=series_id: fetch_fred(s, start, end, timeout, retries))
    for ticker, name in {**dcfg["yahoo_ohlcv"], **dcfg["yahoo_close"]}.items():
        attempt(name, lambda t=ticker: fetch_yahoo(t, start, end, retries))
    if dcfg.get("opec_url"):
        attempt("opec", lambda: fetch_opec(dcfg["opec_url"], timeout))
    for name, msg in status.items():
        log.info("%-12s %s", name, msg)
    return status
