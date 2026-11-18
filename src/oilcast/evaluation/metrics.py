"""Forecast accuracy metrics (price space), interval metrics and the Diebold-Mariano test."""

from __future__ import annotations

import numpy as np
from scipy import stats


def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    mse = float(np.mean(err**2))
    variance = float(np.var(actual))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(mse)),
        "r2": 1.0 - mse / variance if variance > 0 and len(actual) > 1 else float("nan"),
    }


def directional_accuracy(origin_price: np.ndarray, actual: np.ndarray, pred: np.ndarray) -> float:
    """Share of origins where the predicted direction of the move matches the realised one.

    Undefined (NaN) for a forecast that never predicts a move, such as the naive model.
    """
    p = np.sign(pred - origin_price)
    a = np.sign(actual - origin_price)
    mask = p != 0
    return float(np.mean(p[mask] == a[mask])) if mask.any() else float("nan")


def pinball(y: np.ndarray, q: np.ndarray, quantiles: list[float]) -> float:
    """Mean pinball loss of a (n, Q) quantile matrix against (n,) outcomes."""
    tau = np.asarray(quantiles)[None, :]
    diff = y[:, None] - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def interval_metrics(
    actual: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float
) -> dict[str, float]:
    """Empirical coverage, mean width and Winkler (interval) score, all in price units."""
    alpha = 1 - level
    inside = (actual >= lo) & (actual <= hi)
    width = hi - lo
    penalty = (2 / alpha) * (np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0))
    return {
        "coverage": float(np.mean(inside)),
        "width": float(np.mean(width)),
        "winkler": float(np.mean(width + penalty)),
    }


def newey_west_variance(d: np.ndarray, lags: int) -> float:
    """Long-run variance of ``d`` with Bartlett weights (always non-negative)."""
    d = d - d.mean()
    n = len(d)
    var = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * float(d[k:] @ d[:-k]) / n
    return var


def diebold_mariano(
    err_model: np.ndarray, err_base: np.ndarray, h: int, loss: str = "squared"
) -> dict[str, float]:
    """DM test of equal accuracy. Negative statistic = model better than the baseline.

    h-step forecasts made every day overlap, so the loss differential is autocorrelated up
    to lag h-1; the variance is estimated with Newey-West using h-1 lags. With long
    horizons (h=252) on a few years of test data the effective sample is small, so the
    p-value is only indicative.
    """
    f = np.square if loss == "squared" else np.abs
    d = f(err_model) - f(err_base)
    n = len(d)
    var = newey_west_variance(d, max(h - 1, 0))
    if n < 2 or var <= 0:
        return {"dm_stat": float("nan"), "dm_pvalue": float("nan")}
    stat = float(d.mean() / np.sqrt(var / n))
    return {"dm_stat": stat, "dm_pvalue": float(2 * stats.norm.sf(abs(stat)))}
