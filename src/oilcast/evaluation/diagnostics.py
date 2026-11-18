"""Reproducible OHLC candles and paired forecast-error violins from saved observations."""

from __future__ import annotations

import hashlib
import textwrap
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from oilcast.evaluation.metrics import point_metrics
from oilcast.paths import Paths
from oilcast.utils import write_json

MODELS = ["naive", "rnn", "lstm", "gru"]
LABELS = {"naive": "Ingenuo", "rnn": "RNN simple", "lstm": "LSTM", "gru": "GRU"}
COLORS = {"naive": "#9aabb7", "rnn": "#83ebc4", "lstm": "#aaa9f5", "gru": "#efb671"}
HORIZONS = {5: "5 sesiones · 1 semana", 63: "63 sesiones · 3 meses", 252: "252 sesiones · 1 año"}


@contextmanager
def plot_style():
    with plt.rc_context(
        {
            "figure.facecolor": "#10191f",
            "axes.facecolor": "#10191f",
            "axes.edgecolor": "#36464e",
            "axes.labelcolor": "#cad6d7",
            "text.color": "#ecf2ef",
            "xtick.color": "#becbcf",
            "ytick.color": "#becbcf",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "#10191f",
        }
    ):
        yield


def candle_data(paths: Paths, sessions: int = 90) -> tuple[pd.DataFrame, int]:
    """Reject invalid or missing candles; do not invent OHLC for the daily spot series."""
    frame = pd.read_csv(paths.raw / "brent_fut.csv", parse_dates=["date"]).set_index("date")
    if frame.index.duplicated().any():
        raise ValueError("Duplicate futures dates")
    columns = ["open", "high", "low", "close"]
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(values).all(axis=1)
        & (values > 0).all(axis=1)
        & (values["high"] >= values[["open", "close", "low"]].max(axis=1))
        & (values["low"] <= values[["open", "close", "high"]].min(axis=1))
    )
    clean = frame.loc[valid].sort_index().tail(sessions).copy()
    if clean.empty:
        raise ValueError("No valid OHLC observations")
    return clean, int((~valid).sum())


def error_samples(paths: Paths, h: int) -> pd.DataFrame:
    """Identical origins/targets/outcomes for all four models; signed error = forecast-real."""
    frames = []
    reference = None
    for model in MODELS:
        suffix = "" if model == "naive" else "_ensemble"
        path = paths.results / "predictions" / f"{model}_h{h}{suffix}.csv"
        frame = pd.read_csv(path, parse_dates=["origin", "target_date"])
        frame = frame.loc[frame["phase"] == "test"].sort_values("origin").reset_index(drop=True)
        if frame.empty or frame["origin"].duplicated().any():
            raise ValueError(f"Missing or duplicate test origins in {path}")
        if reference is None:
            reference = frame
        elif not frame[["origin", "target_date"]].equals(
            reference[["origin", "target_date"]]
        ) or not np.allclose(frame[["price", "y"]], reference[["price", "y"]], rtol=3e-5):
            raise ValueError("Violin plots require identical origins and realised targets")
        # Use one common outcome array despite six-significant-figure archived CSVs.
        q = np.zeros(len(frame)) if model == "naive" else frame["q50"].to_numpy()
        actual_return = np.expm1(reference["y"].to_numpy())
        predicted_return = np.expm1(q)
        errors = predicted_return - actual_return
        if not np.isfinite(errors).all():
            raise ValueError("Non-finite forecast errors")
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "horizon": h,
                    "origin": reference["origin"],
                    "target_date": reference["target_date"],
                    "error_price": reference["price"].to_numpy() * errors,
                    "error_return_pp": 100 * errors,
                    "actual_price": reference["price"].to_numpy() * (1 + actual_return),
                    "predicted_price": reference["price"].to_numpy() * (1 + predicted_return),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def draw_candles(frame: pd.DataFrame, path) -> None:
    with plot_style():
        fig, (ax, vol) = plt.subplots(
            2,
            1,
            figsize=(14, 6.8),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        )
        for i, (_, row) in enumerate(frame.iterrows()):
            color = "#83ebc4" if row["close"] >= row["open"] else "#efb671"
            ax.vlines(i, row["low"], row["high"], color=color, linewidth=1)
            bottom = min(row["open"], row["close"])
            height = abs(row["close"] - row["open"])
            if height < 1e-8:
                ax.hlines(row["close"], i - 0.32, i + 0.32, color=color, linewidth=1.5)
            else:
                ax.add_patch(
                    Rectangle(
                        (i - 0.32, bottom),
                        0.64,
                        height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.88,
                    )
                )
            volume = row.get("volume", np.nan)
            if np.isfinite(volume) and volume >= 0:
                vol.bar(i, volume / 1000, color=color, width=0.64, alpha=0.6)
        ax.autoscale_view()
        ax.set_xlim(-1, len(frame))
        ax.set_ylabel("USD / barril")
        ax.grid(axis="y", color="#304048", alpha=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.legend(
            handles=[
                Patch(color="#83ebc4", label="Cierre ≥ apertura"),
                Patch(color="#efb671", label="Cierre < apertura"),
            ],
            loc="upper left",
            frameon=False,
            ncol=2,
            fontsize=9,
        )
        vol.set_ylabel("Volumen\n(miles)", fontsize=9)
        ticks = np.unique(np.linspace(0, len(frame) - 1, 7).astype(int))
        vol.set_xticks(ticks, [frame.index[i].strftime("%d/%m/%Y") for i in ticks], fontsize=9)
        vol.set_xlabel("Sesiones observadas · no se rellenan festivos", labelpad=10)
        fig.suptitle("Brent futuros · velas diarias", x=0.075, ha="left", fontsize=20, y=0.98)
        fig.text(
            0.075,
            0.915,
            f"BZ=F · {len(frame)} sesiones · {frame.index[0]:%d/%m/%Y}–{frame.index[-1]:%d/%m/%Y}",
            color="#b9cbc7",
            fontsize=11,
        )
        fig.text(
            0.075,
            0.015,
            "Fuente: Yahoo Finance, serie continua. Mecha: mínimo–máximo; cuerpo: apertura–cierre. "
            "Producto y fechas distintos del Brent spot usado por el modelo.",
            color="#9eafb3",
            fontsize=8,
        )
        fig.subplots_adjust(top=0.86, bottom=0.16, left=0.075, right=0.98)
        fig.savefig(path, dpi=170)
        plt.close(fig)


def draw_violin_axis(ax, frame: pd.DataFrame, unit: str) -> None:
    column = "error_price" if unit == "precio" else "error_return_pp"
    ax.axhline(0, color="#d0dcd8", linewidth=1, linestyle=(0, (3, 4)), alpha=0.7)
    labels = []
    for i, model in enumerate(MODELS, start=1):
        values = frame.loc[frame["model"] == model, column].to_numpy()
        if np.ptp(values) > 1e-12:
            parts = ax.violinplot(
                [values],
                positions=[i],
                widths=0.78,
                showextrema=False,
                points=180,
                bw_method="scott",
            )
            body = parts["bodies"][0]
            body.set_facecolor(COLORS[model])
            body.set_edgecolor(COLORS[model])
            body.set_alpha(0.5)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(i, values.min(), values.max(), color=COLORS[model], linewidth=0.7, alpha=0.6)
        ax.vlines(i, q1, q3, color=COLORS[model], linewidth=5)
        ax.scatter(i, median, color="#f4f8f5", s=19, zorder=3, edgecolors="#17231f", linewidths=0.6)
        labels.append(f"{LABELS[model]}\nMAE {np.mean(np.abs(values)):.2f}")
    ax.set_xticks(range(1, 5), labels, fontsize=9)
    ax.set_xlim(0.4, 4.6)
    ax.grid(axis="y", color="#304048", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Error (USD/barril)" if unit == "precio" else "Error de rentabilidad (p. p.)")
    h = int(frame["horizon"].iloc[0])
    ax.set_title(HORIZONS[h] + f"\nn = {len(frame) // 4} orígenes", fontsize=12, pad=15)


def draw_violins(samples: dict[int, pd.DataFrame], unit: str, path) -> None:
    with plot_style():
        count = len(samples)
        fig, axes = plt.subplots(
            1, count, figsize=(6.4, 7) if count == 1 else (5.3 * count, 6.3), squeeze=False
        )
        for ax, frame in zip(axes.ravel(), samples.values(), strict=True):
            draw_violin_axis(ax, frame, unit)
            if count == 1:
                ax.tick_params(axis="both", labelsize=11)
        name = "precio" if unit == "precio" else "rentabilidad simple"
        title = (
            f"Distribución del error · {name}"
            if count == 1
            else f"Dónde se concentran los errores · {name}"
        )
        fig.suptitle(title, x=0.055, ha="left", fontsize=16 if count == 1 else 18, y=0.98)
        subtitle = "Previsión − observado · positivo: sobreestima · negativo: subestima"
        fig.text(
            0.055,
            0.875 if count == 1 else 0.91,
            textwrap.fill(subtitle, 65) if count == 1 else subtitle,
            fontsize=9 if count == 1 else 10,
            color="#becdca",
            linespacing=1.6,
        )
        footer = (
            "Ancho: densidad suavizada. Punto blanco: mediana. Barra gruesa: cuartiles. "
            "Línea fina: mínimo–máximo.\nEscala vertical propia por panel. "
            "Previsiones diarias solapadas: las observaciones no son independientes."
        )
        fig.text(
            0.055,
            0.025 if count == 1 else 0.045,
            textwrap.fill(footer.replace("\n", " "), 81) if count == 1 else footer,
            fontsize=8,
            color="#9eafb3",
            linespacing=1.6,
        )
        fig.subplots_adjust(
            top=0.78, bottom=0.22, left=0.18 if count == 1 else 0.065, right=0.98, wspace=0.3
        )
        fig.savefig(path, dpi=170)
        plt.close(fig)


def draw_scatter(samples: dict[int, pd.DataFrame], arch: str, path) -> None:
    with plot_style():
        fig, axes = plt.subplots(1, 3, figsize=(15.9, 6))
        for ax, (h, sample) in zip(axes, samples.items(), strict=True):
            frame = sample[sample["model"] == arch]
            actual, predicted = (
                frame["actual_price"].to_numpy(),
                frame["predicted_price"].to_numpy(),
            )
            values = point_metrics(actual, predicted)
            low = min(actual.min(), predicted.min())
            high = max(actual.max(), predicted.max())
            pad = max((high - low) * 0.06, 0.5)
            ax.scatter(actual, predicted, s=9, color=COLORS[arch], alpha=0.35, edgecolors="none")
            ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", color="#d4bd8d", lw=1)
            ax.set(
                xlim=(low - pad, high + pad),
                ylim=(low - pad, high + pad),
                xlabel="Real observado (USD/barril)",
                ylabel="Predicción (USD/barril)",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                HORIZONS[h] + f"\nR² = {values['r2']:.3f} · MAE = {values['mae']:.2f}",
                fontsize=12,
                pad=14,
            )
            ax.grid(color="#304048", alpha=0.4)
            ax.set_axisbelow(True)
        fig.suptitle(
            f"Test frente a realidad · {LABELS[arch]}", x=0.055, ha="left", fontsize=20, y=0.98
        )
        fig.text(
            0.055,
            0.89,
            "Cada punto es una previsión fuera de muestra. "
            "La diagonal representa el acierto perfecto (y = x).",
            fontsize=11,
            color="#b9cbc7",
        )
        fig.text(
            0.055,
            0.035,
            "R² de las predicciones originales: 1 es perfecto; 0 iguala la media del periodo; "
            "negativo es peor que esa media.\n"
            "No se ajusta una recta a los puntos. Periodo ya consultado; previsiones solapadas. "
            "No es una evaluación independiente nueva.",
            fontsize=9,
            color="#a3b6b3",
            linespacing=1.6,
        )
        fig.subplots_adjust(top=0.77, bottom=0.2, left=0.065, right=0.985, wspace=0.32)
        fig.savefig(path, dpi=170)
        plt.close(fig)


def build_diagnostics(paths: Paths) -> dict:
    paths.figures.mkdir(parents=True, exist_ok=True)
    candles, excluded = candle_data(paths)
    draw_candles(candles, paths.figures / "brent_velas.png")
    samples = {h: error_samples(paths, h) for h in HORIZONS}
    for arch in ("rnn", "lstm", "gru"):
        draw_scatter(samples, arch, paths.figures / f"dispersion_{arch}.png")
    for unit in ("precio", "retorno"):
        draw_violins(samples, unit, paths.figures / f"violines_{unit}.png")
        for h, frame in samples.items():
            draw_violins({h: frame}, unit, paths.figures / f"violines_{unit}_h{h}.png")
    inputs = [paths.raw / "brent_fut.csv"] + [
        paths.results / "predictions" / f"{model}_h{h}{'' if model == 'naive' else '_ensemble'}.csv"
        for h in HORIZONS
        for model in MODELS
    ]
    hashes = {
        str(p.relative_to(paths.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
    }
    manifest = {
        "candles": {
            "source": "Yahoo Finance · BZ=F",
            "sessions": len(candles),
            "first": candles.index[0].date().isoformat(),
            "last": candles.index[-1].date().isoformat(),
            "excludedInvalid": excluded,
        },
        "samples": {
            str(h): {
                "n": int(len(frame) // 4),
                "first": frame["origin"].min().date().isoformat(),
                "last": frame["origin"].max().date().isoformat(),
            }
            for h, frame in samples.items()
        },
        "input_sha256": hashes,
        "fingerprint": hashlib.sha256("".join(hashes.values()).encode()).hexdigest()[:16],
    }
    write_json(paths.figures / "diagnostics.json", manifest)
    return manifest
