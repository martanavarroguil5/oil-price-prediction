"""Experimental interval preview without modifying published forecasts or weights."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from oilcast.evaluation.interval_calibration import calibrate_intervals
from oilcast.evaluation.interval_study import load_frame
from oilcast.inference.forecaster import HorizonForecast
from oilcast.paths import Paths
from oilcast.utils import read_json


def interval_preview(paths: Paths, forecasts: dict[int, HorizonForecast], arch: str) -> dict | None:
    folder = paths.reports / "experiments/intervals"
    if not (folder / "app.json").exists():
        return None
    study = read_json(folder / "app.json")
    decision = read_json(folder / "decision.json")
    latest = {}
    for h, fc in forecasts.items():
        if arch not in study["selected"] or str(h) not in study["selected"][arch]:
            continue
        source = paths.results / "predictions" / f"{arch}_h{h}_ensemble.csv"
        key = source.relative_to(paths.root).as_posix()
        expected = decision["input_sha256"].get(key) or decision["input_sha256"].get(
            key.replace("/", "\\")
        )
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            return {**study, "latest": {}, "stale": True}
        name = study["selected"][arch][str(h)]
        candidate = next(c for c in study["candidates"] if c["name"] == name)
        # Issued after the last observed close: only outcomes realised before next day.
        query = {"origin": fc.origin_date + pd.Timedelta(days=1)}
        for quantile, value in fc.quantiles.items():
            query[f"q{round(100 * quantile):02d}"] = np.log(value / fc.last_price)
        for level, (lo, hi) in fc.intervals.items():
            query[f"lo{level}"], query[f"hi{level}"] = (
                np.log(lo / fc.last_price),
                np.log(hi / fc.last_price),
            )
        bounds = calibrate_intervals(
            load_frame(paths, arch, h), pd.DataFrame([query]), candidate, study["min_scores"]
        ).iloc[0]
        latest[str(h)] = {
            "method": name,
            "origin": fc.origin_date.date().isoformat(),
            "target": fc.target_date.date().isoformat(),
            "published": fc.intervals,
            "experimental": {
                str(level): [
                    fc.last_price * float(np.exp(bounds[f"lo{level}"])),
                    fc.last_price * float(np.exp(bounds[f"hi{level}"])),
                ]
                for level in (50, 80)
            },
        }
    return {**study, "latest": latest, "arch": arch, "stale": False}
