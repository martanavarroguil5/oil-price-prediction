"""Evaluate the saved central forecasts in simple returns (percentage points)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oilcast.paths import Paths


def return_scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Inputs are simple returns, not prices or log returns. No division by actual return."""
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    naive_mae = float(np.mean(np.abs(actual)))
    return {
        "mae_pp": 100 * mae,
        "rmse_pp": 100 * float(np.sqrt(np.mean(errors**2))),
        "skill_mae": 1 - mae / naive_mae if naive_mae > 0 else None,
        "n": len(actual),
    }


def return_comparison(paths: Paths) -> list[dict]:
    rows = []
    for h in (5, 63, 252):
        reference = None
        for arch in ("rnn", "lstm", "gru"):
            path = paths.results / "predictions" / f"{arch}_h{h}_ensemble.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            test = frame.loc[frame["phase"] == "test"].sort_values("origin")
            if test.empty:
                continue
            actual = np.expm1(test["y"].to_numpy())
            if reference is None:
                reference = test
                rows.append(
                    {"horizon": h, "model": "naive", **return_scores(actual, np.zeros_like(actual))}
                )
            elif test[["origin", "target_date"]].to_numpy().tolist() != reference[
                ["origin", "target_date"]
            ].to_numpy().tolist() or not np.allclose(test["y"], reference["y"]):
                raise ValueError(
                    "Return comparisons require identical forecast origins and targets"
                )
            rows.append(
                {
                    "horizon": h,
                    "model": arch,
                    **return_scores(actual, np.expm1(test["q50"].to_numpy())),
                }
            )
    return rows
