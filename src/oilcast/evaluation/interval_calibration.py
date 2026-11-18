"""Causal interval calibrators shared by historical evaluation and current forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilcast.evaluation.conformal import conformal_offset, cqr_scores

LEVELS = {50: ("q25", "q75"), 80: ("q10", "q90")}


def calibrate_intervals(
    history: pd.DataFrame, queries: pd.DataFrame, candidate: dict, min_scores: int = 100
) -> pd.DataFrame:
    """Only history.target_date < query.origin is available for each query.

    Returns log-return bounds. ``current`` copies the published bounds exactly. Other
    candidates contain the median and nest 50% inside 80%, including during warm-up.
    Query rows need no realised outcomes, so this same function can forecast live.
    """
    kind, window = candidate["kind"], candidate.get("window")
    if kind not in {"current", "cqr", "scaled", "residual"}:
        raise ValueError("Unknown interval calibrator")
    if min_scores < 1 or (window is not None and window < min_scores):
        raise ValueError("Window must allow the minimum number of scores")
    if kind == "current":
        return queries[["lo50", "hi50", "lo80", "hi80"]].copy().reset_index(drop=True)
    dates = pd.to_datetime(history["target_date"]).to_numpy(dtype="datetime64[ns]")
    origins = pd.to_datetime(queries["origin"]).to_numpy(dtype="datetime64[ns]")
    if np.isnat(dates).any() or np.isnat(origins).any():
        raise ValueError("Calibration dates must be realised dates")
    order = np.argsort(dates, kind="stable")
    counts = np.searchsorted(dates[order], origins, side="left")
    y = history["y"].to_numpy()
    median = queries["q50"].to_numpy()
    out = pd.DataFrame(index=np.arange(len(queries)))
    for level, (lower, upper) in LEVELS.items():
        hlo, hhi = history[lower].to_numpy(), history[upper].to_numpy()
        lo, hi = queries[lower].to_numpy().copy(), queries[upper].to_numpy().copy()
        if not np.isfinite(np.concatenate([y, hlo, hhi, lo, hi, median])).all():
            raise ValueError("Non-finite interval inputs")
        if (hlo > hhi).any() or (lo > hi).any():
            raise ValueError("Unordered raw quantiles")
        hscale = np.maximum((hhi - hlo) / 2, 1e-8)
        qscale = np.maximum((hi - lo) / 2, 1e-8)
        if kind == "residual":
            scores = np.abs(y - history["q50"].to_numpy())
        else:
            scores = cqr_scores(y, hlo, hhi)
            if kind == "scaled":
                scores = scores / hscale
        scores = scores[order]
        for i, count in enumerate(counts):
            start = 0 if window is None else max(0, int(count) - window)
            past = scores[start:count]
            if len(past) < min_scores:
                continue
            correction = conformal_offset(past, level / 100)
            if kind == "residual":
                lo[i], hi[i] = median[i] - correction, median[i] + correction
            else:
                if kind == "scaled":
                    correction *= qscale[i]
                lo[i] -= correction
                hi[i] += correction
        lo, hi = np.minimum(lo, median), np.maximum(hi, median)
        if level == 80:
            lo = np.minimum(lo, out["lo50"].to_numpy())
            hi = np.maximum(hi, out["hi50"].to_numpy())
        out[f"lo{level}"], out[f"hi{level}"] = lo, hi
    out["calibration_n"] = np.minimum(counts, window) if window else counts
    return out
