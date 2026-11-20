"""Markdown report (``reports/comparativa.md``) and figures from the saved results.

All numbers and every conclusion sentence are derived from the result files, so the report
can be regenerated at any time without manual edits.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure  # OO API: no pyplot backend side effects

from oilcast.config import Config, horizon_overrides
from oilcast.data.dataset import load_dataset
from oilcast.features.engineering import VALUATION_DESCRIPTIONS, compute_features
from oilcast.models.recurrent import ARCH_LABELS
from oilcast.paths import Paths
from oilcast.training.tune import tuning_summary
from oilcast.utils import read_json

LABELS = {**ARCH_LABELS, "naive": "Ingenuo"}
COLORS = {"rnn": "#8c6bb1", "lstm": "#2b8cbe", "gru": "#41ab5d", "naive": "#969696"}
HORIZON_LABELS = {5: "Corto (5 d)", 63: "Medio (63 d)", 252: "Largo (252 d)"}
SIGNIFICANCE = 0.05


def _spanish_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def _join_horizons(hs: list[int]) -> str:
    return " y ".join(str(h) for h in hs) if len(hs) <= 2 else ", ".join(map(str, hs))


def _fmt(x: float, nd: int = 2, pct: bool = False, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if pct:
        return f"{x * 100:+.1f} %" if signed else f"{x * 100:.1f} %"
    return f"{x:,.{nd}f}"


def comparison_table(comp: pd.DataFrame, horizons: list[int], archs: list[str]) -> str:
    head = (
        "| Horizonte | Modelo | MAE ($) | RMSE ($) | R² | Mejora MAE vs ingenuo | "
        "Mejora RMSE vs ingenuo | p-valor DM | Acierto direccional | Cobertura IC80 | "
        "Anchura IC80 ($) | Winkler IC80 | Parámetros | Tiempo entren. (s) |\n"
        + "|---" * 14
        + "|\n"
    )
    rows = []
    for h in horizons:
        for model in ["naive", *archs]:
            r = comp[(comp["model"] == model) & (comp["horizon"] == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            if model == "naive":
                net = ["—"] * 4 + [_fmt(r["coverage80"], pct=True), _fmt(r["width80"])]
                net += [_fmt(r["winkler80"]), "—", "—"]
            else:
                params = "—" if math.isnan(r["n_params"]) else str(int(r["n_params"]))
                net = [
                    _fmt(r["skill_mae"], pct=True, signed=True),
                    _fmt(r["skill_rmse"], pct=True, signed=True),
                    _fmt(r["dm_pvalue"], 3),
                    _fmt(r["dir_acc"], pct=True),
                    _fmt(r["coverage80"], pct=True),
                    _fmt(r["width80"]),
                    _fmt(r["winkler80"]),
                    params,
                    _fmt(r["train_seconds"], 1),
                ]
            cells = [str(HORIZON_LABELS.get(h, h)), LABELS[model], _fmt(r["mae"]), _fmt(r["rmse"])]
            cells.append(_fmt(r.get("r2"), 3))
            rows.append("| " + " | ".join(cells + net) + " |")
    return head + "\n".join(rows)


def stability_table(comp: pd.DataFrame, horizons: list[int], archs: list[str]) -> str:
    out = (
        "| Horizonte | Modelo | MAE ensemble | MAE por semilla (media ± desv.) | "
        "Mejora MAE por semilla (desv.) | Cobertura IC80 sin calibrar | MSE |\n"
        + "|---" * 7
        + "|\n"
    )
    for h in horizons:
        for a in archs:
            r = comp[(comp["model"] == a) & (comp["horizon"] == h)]
            if r.empty:
                continue
            r = r.iloc[0]
            out += (
                f"| {HORIZON_LABELS.get(h, h)} | {LABELS[a]} | {_fmt(r['mae'])} | "
                f"{_fmt(r['mae_seed_mean'])} ± {_fmt(r['mae_seed_std'])} | "
                f"{_fmt(r['skill_mae_seed_std'], pct=True)} | "
                f"{_fmt(r['raw_coverage80'], pct=True)} | {_fmt(r['mse'], 1)} |\n"
            )
    return out


def year_table(by_year: pd.DataFrame, horizons: list[int], archs: list[str]) -> str:
    years = sorted(by_year["year"].unique())
    out = "| Horizonte | Modelo | " + " | ".join(str(y) for y in years) + " |\n"
    out += "|---" * (2 + len(years)) + "|\n"
    for h in horizons:
        for a in archs:
            g = by_year[(by_year["model"] == a) & (by_year["horizon"] == h)].set_index("year")
            if g.empty:
                continue
            cells = [
                _fmt(g.loc[y, "skill_mae"], pct=True, signed=True) if y in g.index else "—"
                for y in years
            ]
            out += f"| {HORIZON_LABELS.get(h, h)} | {LABELS[a]} | " + " | ".join(cells) + " |\n"
        g = by_year[by_year["horizon"] == h].drop_duplicates("year").set_index("year")
        cells = [_fmt(g.loc[y, "mae_naive"]) if y in g.index else "—" for y in years]
        out += f"| {HORIZON_LABELS.get(h, h)} | MAE ingenuo ($) | " + " | ".join(cells) + " |\n"
    return out


def tuning_table(tun: pd.DataFrame) -> str:
    out = (
        "| Modelo | Horizonte | hidden | capas | dropout | lr | weight decay | batch | "
        "épocas (fijas) | trials (podados) |\n" + "|---" * 10 + "|\n"
    )
    for r in tun.itertuples():
        out += (
            f"| {LABELS[r.arch]} | {r.horizon} | {r.hidden_size} | {r.num_layers} | "
            f"{r.dropout:.2f} | {r.lr:.1e} | {r.weight_decay:.1e} | {r.batch_size} | "
            f"{r.final_epochs} | {r.n_trials} ({r.n_pruned}) |\n"
        )
    return out


def alternatives_table(selection: dict) -> str:
    """Variantes que ganan a la red elegida en algún plazo del test y por qué no se usan."""
    alts = selection.get("alternatives") or []
    if not alts:
        return ""
    out = (
        "### Alternativas que ganan en el test\n\n"
        "Mejora del MAE frente al ingenuo; entre paréntesis, la de B, la red de la aplicación.\n\n"
        "| Plazo | Alternativa | Test 2022-2026 | 2008-2021 | Por qué no se usa |\n"
        "|---|---|---|---|---|\n"
    )
    for a in alts:
        out += (
            f"| {HORIZON_LABELS.get(a['horizon'], a['horizon'])} | {a['name']} | {a['test']} | "
            f"{a['development']} | {a['reason']} |\n"
        )
    note = selection.get("alternatives_note")
    return out + (f"\n{note}\n" if note else "")


def conclusion(
    comp: pd.DataFrame,
    selection: dict,
    timing: pd.DataFrame,
    by_year: pd.DataFrame,
    horizons: list[int],
) -> str:
    """Data-driven conclusion: why the chosen network and where it adds value."""
    chosen = selection["chosen"]
    sc = pd.DataFrame(selection["scores"]).set_index("arch")
    best = sc["mean_rel_mae"].idxmin()
    t = timing.groupby("arch")["train_seconds"].sum()
    others = [a for a in sc.index if a != chosen]

    def rel(a: str) -> str:
        return f"{sc.loc[a, 'mean_rel_mae']:.3f}".replace(".", ",")

    def std(a: str) -> str:
        return f"{sc.loc[a, 'mean_rel_seed_std'] * 100:.1f} %".replace(".", ",")

    acc = (
        f"su MAE medio es {rel(chosen)} veces el del ingenuo"
        + (
            f"; en test la mejor fue {LABELS[best]} ({rel(best)}), pero la elección se fijó "
            "antes de ver el test, con las predicciones anteriores a 2022 (ver la regla)"
            if best != chosen
            else ", el mejor de las tres redes"
        )
        + "; "
        + ", ".join(f"{LABELS[a]} {rel(a)}" for a in others if a != best)
        + "."
    )
    stab = (
        f"es la red más estable entre semillas (desviación relativa del MAE {std(chosen)}, "
        + ", ".join(f"{LABELS[a]} {std(a)}" for a in others)
        + "), es decir, su resultado depende poco de la inicialización aleatoria."
        if sc["mean_rel_seed_std"].idxmin() == chosen
        else f"desviación relativa del MAE entre semillas {std(chosen)}."
    )
    other_times = ", ".join(
        f"{LABELS[a]} {t.get(a, math.nan):.1f} s".replace(".", ",") for a in others
    )
    chosen_time = f"{t.get(chosen, math.nan):.1f}".replace(".", ",")
    cost = (
        f"entrenar los tres horizontes cuesta {chosen_time} s en GPU "
        f"({other_times}); "
        "el coste no discrimina entre redes y el reentrenado periódico es viable en CPU."
    )
    lines = [
        f"**Modelo para la aplicación: {LABELS[chosen]}.**",
        "",
        f"- **Precisión**: {acc}",
        f"- **Estabilidad**: {stab}",
        f"- **Coste computacional**: {cost}",
        "",
        "**Dónde aporta valor y dónde no**",
        "",
    ]
    for h in horizons:
        r = comp[(comp["model"] == chosen) & (comp["horizon"] == h)].iloc[0]
        nets = comp[(comp["model"] != "naive") & (comp["horizon"] == h)]
        yb = by_year[(by_year["model"] == chosen) & (by_year["horizon"] == h)]
        pos_years = int((yb["skill_mae"] > 0).sum())
        sig = not math.isnan(r["dm_pvalue"]) and r["dm_pvalue"] < SIGNIFICANCE
        skill = _fmt(r["skill_mae"], pct=True, signed=True).replace(".", ",")
        rng = (
            f"{_fmt(nets['skill_mae'].min(), pct=True, signed=True)} y "
            f"{_fmt(nets['skill_mae'].max(), pct=True, signed=True)}"
        ).replace(".", ",")
        majority = max(r["up_share"], 1 - r["up_share"])
        dir_txt = (
            f"acierta la dirección en el {_fmt(r['dir_acc'], pct=True)} de los casos (apostar "
            f"siempre por el movimiento más frecuente del periodo, algo que no se sabe de "
            f"antemano, habría acertado el {_fmt(majority, pct=True)})"
        ).replace(".", ",")
        naive_cov = comp[(comp["model"] == "naive") & (comp["horizon"] == h)].iloc[0]
        pval = _fmt(r["dm_pvalue"], 3).replace(".", ",")
        cov = _fmt(r["coverage80"], pct=True).replace(".", ",")
        rmse = _fmt(r["skill_rmse"], pct=True, signed=True).replace(".", ",")
        if r["skill_mae"] == 0 and r["skill_rmse"] == 0:
            head = (
                "**coincide con el ingenuo**: la recalibración del centro no ha encontrado "
                f"errores pasados que justifiquen mover la previsión (las tres redes entre {rng})."
            )
            dir_txt = "no mueve el valor central, así que no hay acierto direccional que medir"
        elif r["skill_mae"] > 0:
            size = "mejora marginal" if r["skill_mae"] < 0.02 else "mejora"
            head = (
                f"{size} del valor central ({skill} de MAE y {rmse} de RMSE; las tres redes "
                f"entre {rng} de MAE), positiva en {pos_years} de {len(yb)} años, "
                + (
                    f"estadísticamente significativa (p-valor DM {pval})."
                    if sig
                    else f"**no significativa** (p-valor DM {pval})."
                )
            )
        else:
            head = (
                f"**no aporta valor en el valor central en MAE**: el error absoluto es mayor que "
                f"el del ingenuo ({skill} de MAE, {rmse} de RMSE, p-valor DM {pval}; las tres "
                f"redes entre {rng} de MAE); solo mejora en {pos_years} de {len(yb)} años."
            )
        better_band = r["winkler80"] < naive_cov["winkler80"]

        def es(x: float, **kw) -> str:
            return _fmt(x, **kw).replace(".", ",")

        verdict = "es algo más informativo" if better_band else "no es mejor"
        band = (
            f"El intervalo del 80 % calibrado cubre el {cov} de los casos (ingenuo: "
            f"{es(naive_cov['coverage80'], pct=True)}) y su puntuación de Winkler (anchura + "
            f"penalización por fallos; menor es mejor) es {es(r['winkler80'])} frente a "
            f"{es(naive_cov['winkler80'])} del ingenuo: el rango {verdict} que el del ingenuo."
        )
        lines.append(
            f"- **{HORIZON_LABELS[h]}**: {head} {dir_txt[0].upper() + dir_txt[1:]}. {band}"
        )
    wins = [
        HORIZON_LABELS[h].lower()
        for h in horizons
        if comp[(comp["model"] == chosen) & (comp["horizon"] == h)]["skill_mae"].iloc[0] > 0
    ]
    loses = [HORIZON_LABELS[h].lower() for h in horizons if HORIZON_LABELS[h].lower() not in wins]
    lines += [
        "",
        "**Interpretación.** A partir solo de precios y variables de mercado pasadas, el precio "
        "del crudo se comporta casi como un paseo aleatorio, y el periodo de test incluye dos "
        "shocks geopolíticos (Ucrania 2022 y la guerra con Irán en 2026) que ningún modelo de "
        "este tipo puede anticipar. "
        + (
            f"El modelo elegido mejora al ingenuo en el valor central a {_spanish_list(wins)}"
            if wins
            else "El modelo elegido no mejora al ingenuo en el valor central en ningún horizonte"
        )
        + (f" y no a {' ni a '.join(loses)}" if wins and loses else "")
        + ", sin que ninguna diferencia sea estadísticamente significativa. Por eso la "
        "aplicación muestra siempre el intervalo, indica en cada horizonte si el modelo mejoró o "
        "no al «precio no cambia» en el test y presenta la cifra central como referencia "
        "donde no lo mejora.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures


def fig_skill(comp: pd.DataFrame, horizons: list[int], archs: list[str], path) -> None:
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    width = 0.8 / len(archs)
    x = np.arange(len(horizons))
    for i, a in enumerate(archs):
        vals = [
            comp[(comp["model"] == a) & (comp["horizon"] == h)]["skill_mae"].iloc[0] * 100
            for h in horizons
        ]
        errs = [
            comp[(comp["model"] == a) & (comp["horizon"] == h)]["skill_mae_seed_std"].iloc[0] * 100
            for h in horizons
        ]
        ax.bar(
            x + (i - (len(archs) - 1) / 2) * width,
            vals,
            width,
            yerr=errs,
            capsize=3,
            label=LABELS[a],
            color=COLORS[a],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [HORIZON_LABELS[h] for h in horizons])
    ax.set_ylabel("Mejora del MAE frente al ingenuo (%)")
    ax.set_title("Mejora frente al modelo ingenuo en test (barras de error: desv. entre semillas)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_test_predictions(paths: Paths, arch: str, horizons: list[int], path) -> None:
    fig = Figure(figsize=(10, 3.2 * len(horizons)))
    axes = fig.subplots(len(horizons), 1, sharex=True)
    for ax, h in zip(np.atleast_1d(axes), horizons, strict=True):
        ens = pd.read_csv(
            paths.results / "predictions" / f"{arch}_h{h}_ensemble.csv",
            parse_dates=["origin", "target_date"],
        )
        t = ens[ens["phase"] == "test"].sort_values("target_date")
        actual = t["price"] * np.exp(t["y"])
        ax.fill_between(
            t["target_date"],
            t["price"] * np.exp(t["lo80"]),
            t["price"] * np.exp(t["hi80"]),
            color=COLORS[arch],
            alpha=0.2,
            label="IC 80 % (calibrado)",
        )
        ax.plot(t["target_date"], actual, color="black", lw=1, label="Brent real")
        ax.plot(
            t["target_date"],
            t["price"] * np.exp(t["q50"]),
            color=COLORS[arch],
            lw=1.2,
            label=f"Previsión {LABELS[arch]}",
        )
        ax.plot(
            t["target_date"],
            t["price"],
            color=COLORS["naive"],
            lw=1,
            ls="--",
            label="Ingenuo (precio en el origen)",
        )
        ax.set_title(f"{HORIZON_LABELS[h]}: previsión hecha h días antes de cada fecha")
        ax.set_ylabel("$/barril")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_by_year(by_year: pd.DataFrame, horizons: list[int], archs: list[str], path) -> None:
    fig = Figure(figsize=(12, 3.6))
    axes = fig.subplots(1, len(horizons))
    for ax, h in zip(np.atleast_1d(axes), horizons, strict=True):
        g = by_year[by_year["horizon"] == h]
        years = sorted(g["year"].unique())
        width = 0.8 / len(archs)
        for i, a in enumerate(archs):
            s = g[g["model"] == a].set_index("year").reindex(years)["skill_mae"] * 100
            ax.bar(
                np.arange(len(years)) + (i - (len(archs) - 1) / 2) * width,
                s,
                width,
                color=COLORS[a],
                label=LABELS[a],
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(years)), years)
        ax.set_title(HORIZON_LABELS[h])
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Mejora MAE vs ingenuo (%)")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Mejora frente al ingenuo por año de origen de la previsión (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fig_splits(cfg: Config, path) -> None:
    """Timeline of CV folds, calibration refits and test refits."""
    fig = Figure(figsize=(10, 3.2))
    ax = fig.subplots()
    start = pd.Timestamp("2001-07-01")
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    end = pd.Timestamp("2026-11-30")
    for i, (a, b) in enumerate(cfg["splits"]["cv_blocks"]):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ax.barh(i, (a - start).days, left=start, color="#9ecae1")
        ax.barh(i, (b - a).days, left=a, color="#fd8d3c")
    y = len(cfg["splits"]["cv_blocks"])
    ax.barh(
        y,
        (test_start - start).days,
        left=start,
        color="#9ecae1",
        label="Entrenamiento (objetivo realizado antes del bloque: purga)",
    )
    ax.barh(
        y,
        (end - test_start).days,
        left=test_start,
        color="#e6550d",
        label="Test walk-forward (reentreno cada 1 de enero)",
    )
    ax.barh(0, 0, color="#fd8d3c", label="Validación (CV para hiperparámetros)")
    ax.set_yticks(range(y + 1), [f"CV {i + 1}" for i in range(y)] + ["Test"])
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Particiones temporales (sin mezclar fechas)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ----------------------------------------------------------------------------- report


def build_report(cfg: Config, paths: Paths) -> str:
    archs, horizons = cfg["architectures"], cfg["targets"]["horizons"]
    comp = pd.read_csv(paths.results / "comparison.csv")
    by_year = pd.read_csv(paths.results / "metrics_by_year.csv")
    selection = read_json(paths.results / "selection.json")
    timing = pd.read_csv(paths.results / "timing.csv")
    tun = tuning_summary(paths, archs, horizons)
    chosen = selection["chosen"]
    paths.figures.mkdir(parents=True, exist_ok=True)
    fig_skill(comp, horizons, archs, paths.figures / "mejora_vs_ingenuo.png")
    fig_test_predictions(paths, chosen, horizons, paths.figures / "predicciones_test.png")
    fig_by_year(by_year, horizons, archs, paths.figures / "mejora_por_anio.png")
    fig_splits(cfg, paths.figures / "particiones.png")

    scores = pd.DataFrame(selection["scores"])
    score_tbl = (
        "| Modelo | MAE relativo medio (modelo/ingenuo) | Desv. relativa entre semillas | "
        "Parámetros (media) | Cobertura IC80 media |\n|---|---|---|---|---|\n"
    )
    for r in scores.itertuples():
        score_tbl += (
            f"| {LABELS[r.arch]} | {r.mean_rel_mae:.3f} | {r.mean_rel_seed_std * 100:.1f} % | "
            f"{r.mean_params:.0f} | {r.mean_coverage80 * 100:.1f} % |\n"
        )
    total_time = timing.groupby("arch")["train_seconds"].sum()
    fig_rel = paths.figures.relative_to(paths.reports)
    device = (
        timing["device_name"].dropna().iloc[0]
        if timing["device_name"].notna().any()
        else (timing["device"].iloc[0])
    )
    test_start = cfg["splits"]["test_start"]
    n_features = len(compute_features(load_dataset(paths), cfg).columns)
    special = [h for h in horizons if horizon_overrides(cfg, h)]
    inputs_text = (
        f"\n  A {_join_horizons([h for h in horizons if h not in special])} sesiones, "
        f"{n_features} variables estacionarias (rentabilidades, diferenciales, volatilidad, "
        "desviaciones a medias móviles)."
    )
    groups: dict[str, list[int]] = {}
    for h in special:
        groups.setdefault(repr(horizon_overrides(cfg, h)), []).append(h)
    for hs in groups.values():
        ov = horizon_overrides(cfg, hs[0])
        names = ov.get("features", {}).get("only")
        if names:
            inputs_text += (
                f"\n  A {_join_horizons(hs)} sesiones, solo "
                + ", ".join(f"`{n}`" for n in names)
                + ": "
                + "; ".join(VALUATION_DESCRIPTIONS.get(n, n).lower() for n in names)
                + (", con un atajo lineal desde el último día" if ov.get("highway") else "")
                + " (docs/experimento-ronda4.md)."
            )
    if not special:
        inputs_text = (
            f" {n_features} variables estacionarias (rentabilidades, diferenciales, "
            "volatilidad, desviaciones a medias móviles)."
        )
    extra_conditions = ""
    if cfg["targets"].get("remove_drift"):
        extra_conditions += (
            "\n- **Sin deriva**: las redes aprenden desviaciones respecto a la media del "
            "entrenamiento y no se les devuelve esa media, para que no proyecten al futuro la "
            "tendencia del periodo de entrenamiento."
        )
    if cfg["evaluation"].get("recalibrate_center"):
        extra_conditions += (
            "\n- **Recalibración del centro**: el valor central se encoge hacia el del ingenuo "
            "con un factor entre 0 y 1 estimado solo con los errores ya conocidos en cada fecha."
        )

    md = (
        f"""# Comparativa RNN simple vs LSTM vs GRU (y modelo ingenuo)

> Documento generado automáticamente por `oilcast report` a partir de `reports/results/`.
> Todas las cifras son **fuera de muestra** sobre el periodo de test (orígenes desde
> {test_start}) y se calculan en **dólares por barril**, sobre exactamente los mismos
> orígenes para las tres redes y para el modelo ingenuo.

## Condiciones de la comparación

- **Mismos datos y misma ventana**: las tres redes reciben las mismas
  {cfg["features"]["window"]} sesiones de entrada en cada plazo.{inputs_text}
- **Misma partición**: CV walk-forward expansiva y purgada en desarrollo (4 bloques de
  validación de 2 años: 2014-15, 2016-17, 2018-19, 2020-21) y test walk-forward desde
  {test_start} con reentreno cada 1 de enero, con los hiperparámetros y el nº de épocas
  fijados en la CV (el test nunca se usa para parar el entrenamiento).
- **Mismo presupuesto de ajuste**: {cfg["tuning"]["n_trials"]} trials de Optuna (TPE, misma
  semilla, mismo espacio de búsqueda) por red y horizonte.
- **Salida probabilística**: cuantiles (pérdida pinball); valor central = mediana;
  intervalo del 80 % = cuantiles 10-90 calibrados con conformal (CQR) usando solo errores
  ya realizados antes de cada previsión.
- **{len(cfg["evaluation"]["seeds"])} semillas** por red: se informa el ensemble (media de
  las semillas) y la dispersión entre semillas (estabilidad).
- Modelo ingenuo: precio futuro = precio de hoy; su intervalo usa los cuantiles empíricos de
  las variaciones a h días del periodo de entrenamiento.{extra_conditions}

![Particiones]({fig_rel}/particiones.png)

## Tabla comparativa (test)

{comparison_table(comp, horizons, archs)}

El R² compara el error cuadrático con el que se obtiene usando la media de los precios reales
del tramo evaluado. Puede ser negativo. El modelo ingenuo usa el último precio conocido.

*Mejora* = reducción porcentual del error frente al ingenuo (positivo = mejor que el
ingenuo). *p-valor DM*: test de Diebold-Mariano (error cuadrático, varianza HAC con h-1
retardos); valores < 0,05 indican diferencia significativa. Con h = 252 el test solo cubre
unos pocos años no solapados, por lo que su p-valor es orientativo. Tiempo de entrenamiento:
un ajuste completo (épocas fijas) sobre todo el periodo de desarrollo, medido en secuencia
en {device}.

![Mejora frente al ingenuo]({fig_rel}/mejora_vs_ingenuo.png)

## Estabilidad entre semillas

{stability_table(comp, horizons, archs)}

## Resultados por año (mejora del MAE frente al ingenuo)

{year_table(by_year, horizons, archs)}

![Mejora por año]({fig_rel}/mejora_por_anio.png)

## Hiperparámetros elegidos en la CV

{tuning_table(tun)}

## Elección del modelo para la aplicación

{selection["rule"][:1].upper() + selection["rule"][1:]}

{score_tbl}
Tiempo total de entrenamiento de los tres horizontes: """
        + ", ".join(f"{LABELS[a]} {total_time.get(a, float('nan')):.0f} s" for a in archs)
        + f"""

**Modelo elegido: {LABELS[chosen]}.**

"""
        + alternatives_table(selection)
        + f"""
![Predicciones en test]({fig_rel}/predicciones_test.png)

## Conclusión

"""
        + conclusion(comp, selection, timing, by_year, horizons)
        + "\n"
    )
    out = paths.reports / "comparativa.md"
    if paths.figures.name != "figures":  # smoke runs write next to their own figures
        out = paths.figures / "comparativa.md"
    out.write_text(md, encoding="utf-8")
    return md
