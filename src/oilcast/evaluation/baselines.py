"""Frozen, horizon-specific comparison of simple models and the saved W1 networks.

The development choice is written before computing historical test metrics. Outputs live
outside the production results directory. See docs/experimento-baselines.md for the limits
of reusing these development folds and an already inspected historical test period.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oilcast.config import Config
from oilcast.evaluation.metrics import diebold_mariano, point_metrics
from oilcast.features.windows import Panel, Scaler, scaler_rows
from oilcast.paths import Paths
from oilcast.training.splits import refit_blocks
from oilcast.utils import write_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    value: float = 0.0


def candidates(cfg: Config) -> list[Candidate]:
    """Order also defines the preference for simplicity inside the fixed tolerance."""
    settings = cfg["baselines"]
    return [
        Candidate("naive", "naive"),
        *[
            Candidate(f"reversion_{w}", "reversion", float(w))
            for w in settings["mean_reversion_windows"]
        ],
        *[
            Candidate(f"ridge_{alpha:g}", "ridge", float(alpha))
            for alpha in sorted(settings["ridge_alphas"], reverse=True)
        ],
        *[Candidate(arch, "network") for arch in cfg["architectures"]],
    ]


def ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Minimise mean squared error + alpha * ||beta||², without an intercept."""
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    gram = x.T @ x / len(y)
    return np.linalg.solve(gram + alpha * np.eye(x.shape[1]), x.T @ y / len(y))


def ridge_design(panel: Panel, train: np.ndarray, cfg: Config) -> np.ndarray:
    """Last observation plus the trailing window mean; train-only clipping/scaling."""
    scaler = Scaler.fit(
        panel.features, scaler_rows(train), tuple(cfg["features"]["clip_quantiles"])
    )
    z = scaler.transform(panel.features).astype(np.float64)
    cumulative = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(z, axis=0)])
    means = np.zeros_like(z)
    w = panel.window
    means[w - 1 :] = (cumulative[w:] - cumulative[:-w]) / w
    return np.column_stack([z, means])


def simple_predictions(
    panel: Panel,
    h: int,
    train: np.ndarray,
    predict: np.ndarray,
    specs: list[Candidate],
    cfg: Config,
) -> dict[str, np.ndarray]:
    """All fitted parameters use train only; predict may contain unseen future origins."""
    y = panel.target(h)[train]
    design = ridge_design(panel, train, cfg)
    out = {}
    for spec in specs:
        if spec.kind == "naive":
            out[spec.name] = np.zeros(len(predict))
        elif spec.kind == "reversion":
            col = panel.feature_names.index(f"dev_ma{int(spec.value)}")
            signal = -panel.features[:, col].astype(np.float64)
            x = signal[train]
            denom = float(x @ x)
            beta = float(np.clip(x @ y / denom, 0, 1)) if denom > 0 else 0.0
            out[spec.name] = beta * signal[predict]
        elif spec.kind == "ridge":
            centered = y - y.mean() if cfg["targets"].get("remove_drift") else y
            beta = ridge_coefficients(design[train], centered, spec.value)
            out[spec.name] = design[predict] @ beta
    return out


def read_network(path: Path, reference: pd.DataFrame, name: str) -> pd.DataFrame:
    """Require exactly matching origins/targets; never silently intersect the samples."""
    saved = pd.read_csv(path, parse_dates=["origin", "target_date", "refit_date"])
    if saved["origin"].duplicated().any():
        raise ValueError(f"Duplicate origins in {path}")
    saved = saved.set_index("origin").sort_index()
    ref = reference.set_index("origin").sort_index()
    if not saved.index.equals(ref.index):
        raise ValueError(f"Origins differ from the dataset/refit schedule: {path}")
    for col in ("target_date", "refit_date", "phase"):
        if not np.array_equal(saved[col].to_numpy(), ref[col].to_numpy()):
            raise ValueError(f"{col} differs in {path}")
    # Existing CSVs round prices and log returns to six significant figures.
    for col in ("price", "y"):
        if not np.allclose(saved[col], ref[col], rtol=3e-5, atol=1e-5):
            raise ValueError(f"{col} differs in {path}")
    if not np.isfinite(saved["q50"]).all():
        raise ValueError(f"Non-finite network predictions in {path}")
    out = ref.reset_index()
    out["pred_log"] = saved["q50"].to_numpy()
    out["model"] = name
    return out


def development_mask(frame: pd.DataFrame, test_start: str) -> pd.Series:
    cutoff = pd.Timestamp(test_start)
    return (frame["origin"] < cutoff) & (frame["target_date"] < cutoff)


def error_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    price = frame["price"].to_numpy()
    actual = price * np.exp(frame["y"].to_numpy())
    predicted = price * np.exp(frame["pred_log"].to_numpy())
    return predicted - actual, price - actual


def skill_interval(
    model_error: np.ndarray,
    naive_error: np.ndarray,
    h: int,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Paired circular block bootstrap; identical resamples for each candidate."""
    n = len(model_error)
    if n < 2 or samples < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    block = min(h, n)
    nblocks = (n + block - 1) // block
    skills = []
    abs_m, abs_n = np.abs(model_error), np.abs(naive_error)
    for _ in range(samples):
        starts = rng.integers(0, n, nblocks)
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        denominator = abs_n[idx].mean()
        if denominator > 0:
            skills.append(1 - abs_m[idx].mean() / denominator)
    if not skills:
        return float("nan"), float("nan")
    lo, hi = np.quantile(skills, [0.025, 0.975])
    return float(lo), float(hi)


def score(frame: pd.DataFrame, h: int, cfg: Config, uncertainty: bool) -> dict:
    if frame.empty:
        raise ValueError(f"No observations to score at horizon {h}")
    frame = frame.sort_values("origin")
    em, en = error_arrays(frame)
    price = frame["price"].to_numpy()
    actual = price * np.exp(frame["y"].to_numpy())
    metrics = point_metrics(actual, actual + em)
    base_mae = float(np.abs(en).mean())
    metrics.update(
        mae_naive=base_mae,
        skill_mae=1 - metrics["mae"] / base_mae,
        skill_rmse=1 - metrics["rmse"] / float(np.sqrt(np.mean(en**2))),
        n=len(frame),
        first_origin=frame["origin"].min().date().isoformat(),
        last_origin=frame["origin"].max().date().isoformat(),
    )
    if uncertainty:
        settings = cfg["baselines"]
        metrics.update(diebold_mariano(em, en, h, loss="absolute"))
        lo, hi = skill_interval(
            em, en, h, settings["bootstrap_samples"], settings["bootstrap_seed"]
        )
        offset_skills = [
            1 - np.abs(em[offset::h]).mean() / np.abs(en[offset::h]).mean()
            for offset in range(min(h, len(frame)))
            if np.abs(en[offset::h]).mean() > 0
        ]
        metrics.update(
            skill_lo95=lo,
            skill_hi95=hi,
            spaced_skill_min=float(np.min(offset_skills)),
            spaced_skill_median=float(np.median(offset_skills)),
            spaced_skill_max=float(np.max(offset_skills)),
            spaced_n_min=len(frame) // h,
        )
    return metrics


def choose_by_horizon(
    development: pd.DataFrame, specs: list[Candidate], tolerance: float
) -> dict[str, dict]:
    """Choose using development scores only; tie preference is frozen candidate order."""
    order = {spec.name: i for i, spec in enumerate(specs)}
    decisions = {}
    for h, group in development.groupby("horizon", sort=True):
        best_skill = float(group["skill_mae"].max())
        eligible = group[group["skill_mae"] >= best_skill - tolerance].copy()
        eligible["complexity"] = eligible["model"].map(order)
        chosen = eligible.sort_values("complexity").iloc[0]
        decisions[str(h)] = {
            "model": chosen["model"],
            "development_skill_mae": float(chosen["skill_mae"]),
            "best_development_model": group.loc[group["skill_mae"].idxmax(), "model"],
            "best_development_skill_mae": best_skill,
        }
    return decisions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_predictions(cfg: Config, paths: Paths, specs: list[Candidate]) -> pd.DataFrame:
    panel = Panel.load(paths, cfg)
    all_frames, audits = [], []
    for h in cfg["targets"]["horizons"]:
        blocks, references = [], []
        target_dates = panel.target_dates(h)
        for block in refit_blocks(panel, h, cfg):
            if not (target_dates[block.train] < np.datetime64(block.refit_date)).all():
                raise ValueError("Training targets cross the refit cutoff")
            idx = block.predict
            ref = pd.DataFrame(
                {
                    "origin": panel.dates[idx],
                    "target_date": target_dates[idx],
                    "refit_date": block.refit_date,
                    "phase": block.phase,
                    "price": panel.price[idx],
                    "y": panel.target(h)[idx],
                    "horizon": h,
                }
            )
            references.append(ref)
            preds = simple_predictions(panel, h, block.train, idx, specs, cfg)
            for name, pred in preds.items():
                blocks.append(ref.assign(model=name, pred_log=pred))
            audits.append(
                {
                    "horizon": h,
                    "refit_date": block.refit_date,
                    "train_n": len(block.train),
                    "latest_train_target": pd.Timestamp(target_dates[block.train].max()),
                    "predict_n": len(idx),
                }
            )
        reference = pd.concat(references, ignore_index=True)
        for spec in specs:
            if spec.kind == "network":
                path = paths.results / "predictions" / f"{spec.name}_h{h}_ensemble.csv"
                blocks.append(read_network(path, reference, spec.name))
        all_frames.extend(blocks)
        log.info("baselines h=%d: %d origins, %d candidates", h, len(reference), len(specs))
    output = paths.root / cfg["baselines"]["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(output / "training_audit.csv", index=False)
    return pd.concat(all_frames, ignore_index=True)


def report(output: Path, dev: pd.DataFrame, test: pd.DataFrame, decisions: dict) -> None:
    def pct(value: float) -> str:
        return f"{value * 100:+.2f} %"

    lines = [
        "# Modelos sencillos y elección por horizonte",
        "",
        "Comparación retrospectiva con datos y predicciones incluidos en el repositorio. "
        "El periodo desde 2022 ya se había consultado. La selección se guarda antes de "
        "calcular sus métricas, usando solo orígenes y objetivos anteriores a 2022.",
        "",
        "[Protocolo y límites](../../../docs/experimento-baselines.md). "
        "La aplicación conserva su modelo publicado.",
        "",
        "## Candidato elegido en desarrollo, por horizonte",
        "",
        "| Sesiones | Elegido | Mejora desarrollo | MAE histórico ($) | Mejora histórica "
        "| IC95 de la mejora | Mejora histórica RNN de la app |",
        "|---|---|---|---|---|---|---|",
    ]
    for h, decision in decisions.items():
        group = test[test["horizon"] == int(h)].set_index("model")
        row = group.loc[decision["model"]]
        rnn = pct(group.loc["rnn", "skill_mae"]) if "rnn" in group.index else "—"
        lines.append(
            f"| {h} | {decision['model']} | {pct(decision['development_skill_mae'])} "
            f"| {row['mae']:.3f} | {pct(row['skill_mae'])} "
            f"| [{pct(row['skill_lo95'])}, {pct(row['skill_hi95'])}] | {rnn} |"
        )
    lines += [
        "",
        "Una mejora positiva reduce el MAE frente a mantener el precio actual. El IC95 usa "
        "bloques de h previsiones, conserva el emparejamiento con el ingenuo y es orientativo. "
        "No es un intervalo del precio futuro ni corrige comparaciones múltiples. "
        "Si incluye cero, no establece una ventaja estable con este remuestreo.",
        "",
        "## Todos los candidatos",
        "",
        "| Sesiones | Modelo | Mejora desarrollo | MAE histórico ($) | RMSE histórico ($) "
        "| Mejora histórica | IC95 de la mejora | p DM absoluto |",
        "|---|---|---|---|---|---|---|---|",
    ]
    merged = test.merge(
        dev[["model", "horizon", "skill_mae"]],
        on=["model", "horizon"],
        suffixes=("", "_dev"),
        validate="one_to_one",
    )
    for _, row in merged.iterrows():
        pvalue = "—"
        if np.isfinite(row["dm_pvalue"]):
            pvalue = "<0.001" if row["dm_pvalue"] < 0.001 else f"{row['dm_pvalue']:.3f}"
        interval = f"[{pct(row['skill_lo95'])}, {pct(row['skill_hi95'])}]"
        if row["model"] == "naive":
            interval = "—"
        lines.append(
            f"| {row['horizon']} | {row['model']} | {pct(row['skill_mae_dev'])} "
            f"| {row['mae']:.3f} | {row['rmse']:.3f} | {pct(row['skill_mae'])} "
            f"| {interval} | {pvalue} |"
        )
    lines += [
        "",
        "Los p-valores DM de este experimento usan error absoluto (MAE); los del informe "
        "original usan error cuadrático. No son directamente intercambiables.",
        "",
        "## Mejor referencia sencilla en el contraste histórico (exploratorio)",
        "",
        "Esta tabla elige después de observar el periodo desde 2022; no cambia la decisión "
        "tomada en desarrollo ni propone un nuevo modelo de producción.",
        "",
        "| Sesiones | Mejor modelo sencillo histórico | Mejora histórica "
        "| Mejora en desarrollo de ese modelo |",
        "|---|---|---|---|",
    ]
    simple = merged[~merged["model"].isin(["rnn", "lstm", "gru"])]
    for h, group in simple.groupby("horizon", sort=True):
        row = group.loc[group["mae"].idxmin()]
        lines.append(
            f"| {h} | {row['model']} | {pct(row['skill_mae'])} | {pct(row['skill_mae_dev'])} |"
        )
    lines += [
        "",
        "## Sensibilidad al solapamiento de los candidatos elegidos",
        "",
        "| Sesiones | Modelo | Mínimo por desplazamiento | Mediana | Máximo "
        "| Mínimo de observaciones por submuestra |",
        "|---|---|---|---|---|---|",
    ]
    for h, decision in decisions.items():
        row = test[(test["horizon"] == int(h)) & (test["model"] == decision["model"])].iloc[0]
        lines.append(
            f"| {h} | {decision['model']} | {pct(row['spaced_skill_min'])} "
            f"| {pct(row['spaced_skill_median'])} | {pct(row['spaced_skill_max'])} "
            f"| {row['spaced_n_min']} |"
        )
    lines += [
        "",
        "Cada submuestra toma un origen cada h observaciones para un desplazamiento distinto. "
        "Este rango muestra sensibilidad; no constituye un intervalo de confianza. "
        "Los huecos del calendario pueden separar todavía más algunos orígenes.",
        "",
        "## Archivos para revisar",
        "",
        "- `decision.json`: selección en desarrollo, regla y hashes de las entradas.",
        "- `development.csv` y `test.csv`: todas las métricas por candidato y plazo.",
        "- `annual.csv`: resultados por año del origen, con muestras emparejadas.",
        "- `predictions.csv`: predicciones individuales reproducibles.",
        "- `training_audit.csv`: cortes y último objetivo utilizado en cada entrenamiento.",
        "",
        "Los ganadores históricos no se sustituyen por los elegidos en desarrollo. "
        "La confirmación independiente sigue pendiente de nuevas predicciones y resultados.",
    ]
    (output / "resultado.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baselines(cfg: Config, paths: Paths) -> dict:
    specs = candidates(cfg)
    output = paths.root / cfg["baselines"]["output_dir"]
    if output.resolve() == paths.results.resolve():
        raise ValueError("Baseline outputs must be separate from production results")
    frame = build_predictions(cfg, paths, specs)
    cutoff = cfg["splits"]["test_start"]
    dev_rows = []
    for (h, name), group in frame.groupby(["horizon", "model"], sort=True):
        dev = group[development_mask(group, cutoff)]
        dev_rows.append({"horizon": h, "model": name, **score(dev, h, cfg, False)})
    development = pd.DataFrame(dev_rows)
    development.to_csv(output / "development.csv", index=False)
    decisions = choose_by_horizon(development, specs, cfg["baselines"]["simplicity_tolerance"])
    files = [paths.dataset_csv, paths.root / "docs/experimento-baselines.md"]
    files += [
        paths.root / relative
        for relative in (
            "src/oilcast/evaluation/baselines.py",
            "src/oilcast/evaluation/metrics.py",
            "src/oilcast/features/windows.py",
            "src/oilcast/features/engineering.py",
            "src/oilcast/training/splits.py",
        )
    ]
    files += [
        paths.results / "predictions" / f"{s.name}_h{h}_ensemble.csv"
        for s in specs
        if s.kind == "network"
        for h in cfg["targets"]["horizons"]
    ]
    decision = {
        "protocol_date": "2026-11-18",
        "status": "retrospective_experiment_not_production",
        "rule": (
            "development MAE per horizon; fixed simplicity preference within "
            f"{cfg['baselines']['simplicity_tolerance']} relative MAE"
        ),
        "test_start": cutoff,
        "selected": decisions,
        "config": cfg,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "input_sha256": {str(p.relative_to(paths.root)): sha256(p) for p in files},
    }
    # Persist the development choice BEFORE calculating any test metrics.
    write_json(output / "decision.json", decision)
    log.info("development choice saved: %s", {h: d["model"] for h, d in decisions.items()})
    test_frame = frame[frame["origin"] >= pd.Timestamp(cutoff)]
    test_rows, annual_rows = [], []
    for (h, name), group in test_frame.groupby(["horizon", "model"], sort=True):
        test_rows.append({"horizon": h, "model": name, **score(group, h, cfg, True)})
        for year, yearly in group.groupby(group["origin"].dt.year):
            annual_rows.append(
                {"horizon": h, "model": name, "year": year, **score(yearly, h, cfg, False)}
            )
    test = pd.DataFrame(test_rows)
    test.to_csv(output / "test.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(output / "annual.csv", index=False)
    frame.to_csv(output / "predictions.csv", index=False, float_format="%.10g")
    report(output, development, test, decisions)
    return decision
