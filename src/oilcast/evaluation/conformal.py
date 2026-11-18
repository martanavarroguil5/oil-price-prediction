"""Online conformalised quantile regression (CQR) for the prediction intervals.

For an interval built from quantiles (lo, hi), the conformity score of a past forecast is
``max(lo - y, y - hi)`` (positive when the outcome fell outside). At each forecast origin
the interval is widened (or narrowed) by the empirical ``level`` quantile of the scores of
the past out-of-sample forecasts whose outcome was already known *strictly
before* that origin. No future information is used.
"""

from __future__ import annotations

import numpy as np


def cqr_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def conformal_offset(scores: np.ndarray, level: float) -> float:
    """Finite-sample corrected ``level`` quantile of the scores."""
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


def online_shrink(
    origins: np.ndarray,
    target_dates: np.ndarray,
    pred: np.ndarray,
    y: np.ndarray,
    min_scores: int,
) -> np.ndarray:
    """Shrinkage factor of the central forecast for each origin, in [0, 1].

    ``lambda = sum(pred * y) / sum(pred^2)`` (least squares through the origin) over the past
    forecasts whose outcome was known strictly before the origin, clipped to [0, 1]. With
    fewer than ``min_scores`` known outcomes it is 0: the forecast falls back to the naive one
    until the network has shown out of sample that its bets pay off.
    """
    order = np.argsort(target_dates, kind="stable")
    td = target_dates[order]
    cum_py = np.concatenate([[0.0], np.cumsum(pred[order] * y[order])])
    cum_pp = np.concatenate([[0.0], np.cumsum(pred[order] ** 2)])
    k = np.searchsorted(td, origins, side="left")  # outcomes known before each origin
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(cum_pp[k] > 0, cum_py[k] / cum_pp[k], 0.0)
    return np.where(k >= min_scores, np.clip(lam, 0.0, 1.0), 0.0)


def online_offsets(
    origins: np.ndarray,
    target_dates: np.ndarray,
    scores: np.ndarray,
    level: float,
    window: int | None,
    min_scores: int,
) -> np.ndarray:
    """Offset for each origin using only scores realised before it (0 if too few).

    ``window`` keeps only the most recent realised scores; ``None`` uses all of them.
    """
    order = np.argsort(target_dates, kind="stable")
    td_sorted = target_dates[order]
    sc_sorted = scores[order]
    offsets = np.zeros(len(origins))
    for i, o in enumerate(origins):
        k = int(np.searchsorted(td_sorted, o, side="left"))  # target_date < origin
        past = sc_sorted[0 if window is None else max(0, k - window) : k]
        if len(past) >= min_scores:
            offsets[i] = conformal_offset(past, level)
    return offsets
