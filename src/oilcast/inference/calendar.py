"""Approximate future Brent trading days.

The EIA Brent spot series skips weekends and, essentially, the UK bank holidays (New Year,
Good Friday, Easter Monday, early-May, spring and summer bank holidays, Christmas and
Boxing Day). Horizons are counted in those trading days, so the forecast target dates are
computed on the same calendar. Dates are approximate (one-off holidays are not modelled).
"""

from __future__ import annotations

import pandas as pd
from dateutil.easter import easter


def _observed(d: pd.Timestamp, taken: set[pd.Timestamp]) -> pd.Timestamp:
    while d.dayofweek >= 5 or d in taken:
        d += pd.Timedelta(days=1)
    return d


def uk_holidays(year: int) -> list[pd.Timestamp]:
    e = pd.Timestamp(easter(year))
    may = pd.date_range(f"{year}-05-01", f"{year}-05-31")
    aug = pd.date_range(f"{year}-08-01", f"{year}-08-31")
    taken: set[pd.Timestamp] = set()
    fixed = []
    for d in (pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 25), pd.Timestamp(year, 12, 26)):
        obs = _observed(d, taken)
        taken.add(obs)
        fixed.append(obs)
    return [
        *fixed,
        e - pd.Timedelta(days=2),
        e + pd.Timedelta(days=1),
        may[may.dayofweek == 0][0],
        may[may.dayofweek == 0][-1],
        aug[aug.dayofweek == 0][-1],
    ]


def add_trading_days(start: pd.Timestamp, n: int) -> pd.Timestamp:
    """The n-th Brent trading day after ``start``."""
    years = range(start.year, start.year + n // 200 + 2)
    holidays = [d for y in years for d in uk_holidays(y)]
    return pd.Timestamp(start + pd.offsets.CustomBusinessDay(n, holidays=holidays))
