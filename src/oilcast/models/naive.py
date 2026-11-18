"""Naive (random-walk) baseline: tomorrow's price = today's price.

Point forecast: zero log return (price unchanged). Its uncertainty band uses the empirical
quantiles of past h-day log returns observed in the training window only.
"""

from __future__ import annotations

import numpy as np


def naive_quantiles(train_returns: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Return the quantile vector (in log-return space) for the naive model.

    The median is forced to 0 so the point forecast is exactly "no change".
    """
    r = train_returns[np.isfinite(train_returns)]
    q = np.quantile(r, quantiles)
    q[np.isclose(quantiles, 0.5)] = 0.0
    return np.sort(q)
