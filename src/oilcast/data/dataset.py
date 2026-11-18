"""Build the modelling dataset on the real Brent calendar.

Rules (see notebook 01 for the evidence behind them):

* The calendar uses Brent observation dates with a recorded spot price, not publication
  timestamps. The target is never filled. The latest observations may be absent because
  of publication lag; historical release timestamps and vintages are not reconstructed.
* Other series carry forward the last value dated on or before each Brent observation.
  There is no backward fill. This is observation-date alignment, not proof of intraday
  availability of every source at the forecast issue time.
* Values unsuitable for the chosen log features (including the genuine negative WTI
  observation on 2020-04-20, non-positive futures prices and zero volume) become NaN.
  Genuine Brent extremes (2008, 2020, 2022, 2026) are kept: they are real prices.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from oilcast.config import Config
from oilcast.paths import Paths
from oilcast.utils import write_json

log = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]


def load_raw(paths: Paths, name: str) -> pd.DataFrame | None:
    path = paths.raw / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


def align_to_calendar(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Last observation dated on or before each date; does not model release timestamps."""
    s = series.dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(calendar)).ffill().reindex(calendar)


def _raw_columns(paths: Paths, cfg: Config) -> dict[str, pd.Series]:
    """Collect raw series as {column_name: series}, unfilled."""
    dcfg = cfg["data"]
    cols: dict[str, pd.Series] = {}
    for name in dcfg["fred"].values():
        df = load_raw(paths, name)
        if df is None:
            raise FileNotFoundError(f"missing raw series {name}; run `oilcast download`")
        cols[name] = df[name]
    for name in dcfg["yahoo_ohlcv"].values():
        df = load_raw(paths, name)
        if df is None:
            raise FileNotFoundError(f"missing raw series {name}; run `oilcast download`")
        for c in [*PRICE_COLS, "volume"]:
            cols[f"{name}_{c}"] = df[c]
    for name in dcfg["yahoo_close"].values():
        df = load_raw(paths, name)
        if df is None:
            raise FileNotFoundError(f"missing raw series {name}; run `oilcast download`")
        cols[name] = df["close"]
    opec = load_raw(paths, "opec")
    if opec is not None:
        cols["opec"] = opec["opec"]
    return cols


def clean_raw(cols: dict[str, pd.Series], cfg: Config) -> tuple[dict[str, pd.Series], dict]:
    """Replace impossible values with NaN (before alignment). Returns cleaned series + log."""
    report: dict[str, list] = {}
    positive = ["brent", "wti", "gas", "copper", "gold", "dxy", "opec"]
    for name in cfg["data"]["yahoo_ohlcv"].values():
        positive += [f"{name}_{c}" for c in PRICE_COLS]
    out = dict(cols)
    for col in positive:
        if col not in out:
            continue
        bad = out[col] <= 0
        if bad.any():
            report[col] = [(d.date().isoformat(), float(v)) for d, v in out[col][bad].items()]
            out[col] = out[col].mask(bad)
    for name in cfg["data"]["yahoo_ohlcv"].values():
        col = f"{name}_volume"
        zero = out[col] <= 0
        if zero.any():
            report[col] = [f"{int(zero.sum())} días con volumen 0 -> NaN"]
            out[col] = out[col].mask(zero)
    return out, report


def build_dataset(cfg: Config, paths: Paths) -> pd.DataFrame:
    """Create ``data/processed/brent_dataset.csv`` on the Brent calendar."""
    cols, report = clean_raw(_raw_columns(paths, cfg), cfg)
    brent = cols["brent"].dropna()
    # end_date (optional) freezes the sample so variants built later stay comparable
    calendar = brent.loc[cfg["data"]["start_date"] : cfg["data"].get("end_date")].index
    df = pd.DataFrame(index=calendar)
    df.index.name = "date"
    df["brent"] = brent.reindex(calendar)
    for col, s in cols.items():
        if col != "brent":
            df[col] = align_to_calendar(s, calendar)

    first_valid = {c: df[c].first_valid_index() for c in df.columns}
    paths.processed.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.dataset_csv, float_format="%.6g")
    write_json(
        paths.cleaning_log,
        {
            "rows": len(df),
            "first_date": calendar[0].date(),
            "last_date": calendar[-1].date(),
            "non_positive_or_zero_volume": report,
            "first_valid": {
                c: (d.date() if d is not None else None) for c, d in first_valid.items()
            },
            "missing_before_first_valid": {
                c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()
            },
        },
    )
    log.info("dataset: %d rows %s -> %s", len(df), calendar[0].date(), calendar[-1].date())
    return df


def load_dataset(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.dataset_csv, index_col="date", parse_dates=["date"])
    return df.astype(np.float64)
