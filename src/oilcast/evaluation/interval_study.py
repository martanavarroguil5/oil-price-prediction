"""Frozen development-only interval selection; retrospective test is reported unchanged."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from oilcast.config import Config
from oilcast.evaluation.baselines import skill_interval
from oilcast.evaluation.interval_calibration import calibrate_intervals
from oilcast.evaluation.metrics import interval_metrics
from oilcast.paths import Paths
from oilcast.utils import write_json

ARCHS = ("rnn", "lstm", "gru")
HORIZONS = (5, 63, 252)


def load_frame(paths: Paths, arch: str, h: int) -> pd.DataFrame:
    frame = (
        pd.read_csv(
            paths.results / "predictions" / f"{arch}_h{h}_ensemble.csv",
            parse_dates=["origin", "target_date"],
        )
        .sort_values("origin")
        .reset_index(drop=True)
    )
    if frame["origin"].duplicated().any():
        raise ValueError("Duplicate forecast origins")
    return frame


def winkler_scores(frame: pd.DataFrame, level: int = 80) -> np.ndarray:
    price = frame["price"].to_numpy()
    actual = price * np.exp(frame["y"].to_numpy())
    lo, hi = (price * np.exp(frame[f"{side}{level}"].to_numpy()) for side in ("lo", "hi"))
    alpha = 1 - level / 100
    return hi - lo + 2 / alpha * (np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0))


def metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("Empty interval evaluation sample")
    price = frame["price"].to_numpy()
    actual = price * np.exp(frame["y"].to_numpy())
    result = {"n": len(frame)}
    for level in (50, 80):
        bounds = [price * np.exp(frame[f"{side}{level}"].to_numpy()) for side in ("lo", "hi")]
        values = interval_metrics(actual, *bounds, level / 100)
        result.update({f"{key}{level}": value for key, value in values.items()})
    return result


def choose_interval(development: pd.DataFrame, settings: dict) -> str:
    """No test data are accepted. Fixed guards and tie preference use the frozen grid."""
    table = development.set_index("candidate")
    current = table.loc["current"]
    tolerance = settings["coverage_tolerance"]
    valid = table[
        (table.index != "current")
        & (table["coverage80"] >= 0.8 - tolerance)
        & (table["coverage50"] >= 0.5 - tolerance)
        & (table["winkler80"] <= current["winkler80"] * (1 - settings["min_skill80"]))
        & (table["winkler50"] <= current["winkler50"] * settings["max_winkler50_ratio"])
    ]
    if valid.empty:
        return "current"
    contenders = valid[valid["winkler80"] <= valid["winkler80"].min() * 1.005]
    order = [c["name"] for c in settings["candidates"]]
    return next(name for name in order if name in contenders.index)


def development_mask(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    cutoff = pd.Timestamp(cfg["splits"]["test_start"])
    return (
        (frame["origin"] >= pd.Timestamp(cfg["interval_study"]["development_start"]))
        & (frame["origin"] < cutoff)
        & (frame["target_date"] < cutoff)
    )


def candidate_frame(frame: pd.DataFrame, candidate: dict, min_scores: int) -> pd.DataFrame:
    out = frame.copy()
    calibrated = calibrate_intervals(frame, frame, candidate, min_scores)
    for col in calibrated:
        out[col] = calibrated[col].to_numpy()
    out["interval_method"] = candidate["name"]
    if candidate["kind"] != "current":
        out = out.drop(columns=["offset50", "offset80"], errors="ignore")
    return out


def build_report(output: Path, selected: pd.DataFrame, decisions: dict) -> None:
    lines = [
        "# Mejora experimental de los intervalos",
        "",
        "[Protocolo](../../../docs/experimento-intervalos.md). Decisiones por red y horizonte "
        "con datos de desarrollo; el periodo posterior ya se había consultado. Las medianas "
        "no cambian. No es un test independiente nuevo ni una sustitución automática "
        "de producción.",
        "",
        "| Red | Sesiones | Elegido en desarrollo | Cobertura 80 % actual → elegido "
        "| Anchura actual → elegida ($) | Winkler actual → elegido | Mejora Winkler80 "
        "| IC95 orientativo |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in selected.itertuples():
        lines.append(
            f"| {r.arch} | {r.horizon} | {r.candidate} | "
            f"{100 * r.current_coverage80:.1f} % → {100 * r.coverage80:.1f} % | "
            f"{r.current_width80:.2f} → {r.width80:.2f} | "
            f"{r.current_winkler80:.2f} → {r.winkler80:.2f} | {100 * r.skill80:+.2f} % | "
            f"[{100 * r.skill_lo95:+.2f} %, {100 * r.skill_hi95:+.2f} %] |"
        )
    lines += [
        "",
        "Positivo significa menor puntuación de Winkler, no menor MAE de la mediana. "
        "El IC95 usa bloques de h previsiones; es orientativo, con pocos bloques a un año, "
        "y no corrige exploración ni comparaciones múltiples.",
        "",
        "La cobertura nominal es del 80 %; más cobertura no equivale siempre a un mejor "
        "intervalo. El criterio de selección incorpora también cobertura y Winkler del 50 %.",
        "",
        "## Archivos",
        "",
        "- `decision.json`: variantes elegidas antes de calcular test, configuración y hashes.",
        "- `development.csv` y `test.csv`: todos los candidatos.",
        "- `selected.csv`: actual frente a elegido y comparación con el ingenuo.",
        "- `annual.csv`: comparación por año del origen.",
        "- `predictions/`: intervalos elegidos, con medianas originales intactas.",
        "- `app.json`: resumen para la comparación experimental en la aplicación.",
        "",
        "La aplicación mantiene identificada la calibración publicada. La opción "
        "experimental no demuestra una garantía de cobertura futura. Si una variante "
        "seleccionada en desarrollo empeora después, ese resultado se conserva y se muestra.",
        "",
        "Selección: " + json.dumps(decisions, ensure_ascii=False),
    ]
    (output / "resultado.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_interval_study(cfg: Config, paths: Paths) -> dict:
    settings = cfg["interval_study"]
    output = paths.root / settings["output_dir"]
    if output.resolve() == paths.results.resolve():
        raise ValueError("Use a separate interval experiment output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    frames, candidates = {}, settings["candidates"]
    dev_rows, decisions = [], {}
    for arch in ARCHS:
        for h in HORIZONS:
            source = load_frame(paths, arch, h)
            mask = development_mask(source, cfg)
            for candidate in candidates:
                frame = candidate_frame(source, candidate, settings["min_scores"])
                frames[arch, h, candidate["name"]] = frame
                dev_rows.append(
                    {
                        "arch": arch,
                        "horizon": h,
                        "candidate": candidate["name"],
                        **metrics(frame.loc[mask]),
                    }
                )
            dev = pd.DataFrame(dev_rows)
            group = dev[(dev["arch"] == arch) & (dev["horizon"] == h)]
            decisions.setdefault(arch, {})[str(h)] = choose_interval(group, settings)
            logging.getLogger(__name__).info(
                "development %s h=%s: %s", arch, h, decisions[arch][str(h)]
            )
    development = pd.DataFrame(dev_rows)
    development.to_csv(output / "development.csv", index=False)
    inputs = [
        paths.root / p
        for p in (
            "docs/experimento-intervalos.md",
            "src/oilcast/evaluation/interval_calibration.py",
            "src/oilcast/evaluation/interval_study.py",
            "src/oilcast/evaluation/metrics.py",
            "src/oilcast/evaluation/conformal.py",
            "src/oilcast/evaluation/baselines.py",
        )
    ]
    inputs += [
        paths.results / "predictions" / f"{a}_h{h}_ensemble.csv" for a in ARCHS for h in HORIZONS
    ]
    inputs += [paths.results / "predictions" / f"naive_h{h}.csv" for h in HORIZONS]
    decision = {
        "protocol_date": "2026-11-18",
        "selected": decisions,
        "config": cfg,
        "status": "experimental_retrospective",
        "input_sha256": {
            p.relative_to(paths.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
        },
    }
    # Freeze development choices before calculating a single test metric.
    write_json(output / "decision.json", decision)
    test_rows, selected_rows, annual_rows = [], [], []
    cutoff = pd.Timestamp(cfg["splits"]["test_start"])
    for arch in ARCHS:
        for h in HORIZONS:
            chosen = decisions[arch][str(h)]
            current = frames[arch, h, "current"]
            current = current[current["origin"] >= cutoff]
            original = metrics(current)
            naive = pd.read_csv(
                paths.results / "predictions" / f"naive_h{h}.csv",
                parse_dates=["origin", "target_date"],
            )
            naive = naive[naive["phase"] == "test"].sort_values("origin").reset_index(drop=True)
            for level, q in ((50, ("q25", "q75")), (80, ("q10", "q90"))):
                naive[f"lo{level}"], naive[f"hi{level}"] = naive[q[0]], naive[q[1]]
            if not np.array_equal(naive["origin"], current["origin"]):
                raise ValueError("Naive and model test origins differ")
            naive_metrics = metrics(naive)
            for candidate in candidates:
                frame = frames[arch, h, candidate["name"]]
                test = frame[frame["origin"] >= cutoff]
                values = metrics(test)
                values["skill80"] = 1 - values["winkler80"] / original["winkler80"]
                test_rows.append(
                    {"arch": arch, "horizon": h, "candidate": candidate["name"], **values}
                )
                if candidate["name"] != chosen:
                    continue
                lo, hi = skill_interval(
                    winkler_scores(test),
                    winkler_scores(current),
                    h,
                    settings["bootstrap_samples"],
                    settings["bootstrap_seed"],
                )
                selected_rows.append(
                    {
                        "arch": arch,
                        "horizon": h,
                        "candidate": chosen,
                        **values,
                        **{f"current_{k}": v for k, v in original.items()},
                        "naive_winkler80": naive_metrics["winkler80"],
                        "skill_vs_naive": 1 - values["winkler80"] / naive_metrics["winkler80"],
                        "skill_lo95": lo,
                        "skill_hi95": hi,
                    }
                )
                frame.to_csv(
                    output / "predictions" / f"{arch}_h{h}.csv", index=False, float_format="%.10g"
                )
                for year, yearly in test.groupby(test["origin"].dt.year):
                    original_year = current[current["origin"].dt.year == year]
                    selected_year = metrics(yearly)
                    current_year = metrics(original_year)
                    annual_rows.append(
                        {
                            "arch": arch,
                            "horizon": h,
                            "year": year,
                            "candidate": chosen,
                            **selected_year,
                            "current_winkler80": current_year["winkler80"],
                            "skill80": 1 - selected_year["winkler80"] / current_year["winkler80"],
                        }
                    )
    selected = pd.DataFrame(selected_rows)
    pd.DataFrame(test_rows).to_csv(output / "test.csv", index=False)
    selected.to_csv(output / "selected.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(output / "annual.csv", index=False)
    build_report(output, selected, decisions)
    write_json(
        output / "app.json",
        {
            "status": "experimental_retrospective",
            "selected": decisions,
            "candidates": candidates,
            "min_scores": settings["min_scores"],
            "comparison": selected.to_dict(orient="records"),
        },
    )
    return decision
