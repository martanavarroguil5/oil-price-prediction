"""Replay saved out-of-sample forecasts without returning outcomes before reveal."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from oilcast.data.dataset import load_dataset
from oilcast.paths import Paths


def replay_frame(paths: Paths, arch: str, horizon: int) -> pd.DataFrame:
    if arch not in {"rnn", "lstm", "gru"} or horizon not in {5, 63, 252}:
        raise ValueError("Modelo u horizonte no válido")
    frame = pd.read_csv(
        paths.results / "predictions" / f"{arch}_h{horizon}_ensemble.csv",
        parse_dates=["origin", "target_date", "refit_date"],
    )
    return frame.loc[frame["phase"] == "test"].sort_values("origin").set_index("origin")


def replay_dates(paths: Paths, arch: str, horizon: int) -> dict:
    frame = replay_frame(paths, arch, horizon)
    return {"origins": frame.index.strftime("%Y-%m-%d").tolist()}


def replay_case(paths: Paths, arch: str, horizon: int, origin: str, reveal: bool = False) -> dict:
    """The initial response contains neither the realised target nor future observations."""
    if date.fromisoformat(origin).isoformat() != origin:
        raise ValueError("Fecha no válida")
    frame = replay_frame(paths, arch, horizon)
    day = pd.Timestamp(origin)
    if day not in frame.index:
        raise ValueError("No hay una previsión registrada para esa fecha")
    row = frame.loc[day]
    price = float(row["price"])
    median = float(price * np.exp(row["q50"]))
    lo = float(price * np.exp(row["lo80"]))
    hi = float(price * np.exp(row["hi80"]))
    history = load_dataset(paths)["brent"]
    past = history.loc[:day].tail(252)

    def points(series: pd.Series) -> list:
        return [[d.date().isoformat(), float(p)] for d, p in series.items()]

    result = {
        "origin": origin,
        "targetDate": row["target_date"].date().isoformat(),
        "refitDate": row["refit_date"].date().isoformat(),
        "model": arch,
        "horizon": horizon,
        "price": price,
        "median": median,
        "interval": [lo, hi],
        "history": points(past),
        "revealed": reveal,
    }
    if reveal:
        actual = float(price * np.exp(row["y"]))
        result["outcome"] = {
            "actual": actual,
            "modelError": abs(median - actual),
            "naiveError": abs(price - actual),
            "insideInterval": lo <= actual <= hi,
            "future": points(history.loc[day : row["target_date"]]),
        }
    return result
