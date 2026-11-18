"""Plain-Spanish explanation of a forecast, generated with deterministic rules.

Reliability statements are derived from the saved out-of-sample test metrics of the
production model (skill vs naive, Diebold-Mariano p-value, interval coverage), never
hard-coded, so they stay truthful after every re-evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from oilcast.inference.forecaster import HorizonForecast

HORIZON_NAMES = {
    5: "corto plazo (1 semana)",
    63: "medio plazo (unos 3 meses)",
    252: "largo plazo (unos 12 meses)",
}
SIGNIFICANCE = 0.05
MATERIAL_SKILL = 0.02  # below +2 % MAE reduction we do not call it a real improvement


@dataclass
class Explanation:
    summary: str
    reliability: str
    reliability_level: str  # "aporta valor" | "mejora no concluyente" | "no mejora al ingenuo"
    risks: list[str]


def _money(x: float) -> str:
    return f"{x:,.2f} $".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(x: float, signed: bool = True) -> str:
    s = f"{x * 100:+.1f} %" if signed else f"{x * 100:.1f} %"
    return s.replace(".", ",")


def _date(d) -> str:
    return d.strftime("%d/%m/%Y")


def reliability_text(
    h: int, metrics: dict, naive: dict, skill_by_year: dict[int, float], arch_label: str
) -> tuple[str, str]:
    skill, p = metrics["skill_mae"], metrics.get("dm_pvalue", float("nan"))
    period = f"{metrics['first_origin'][:4]}-{metrics['last_origin'][:4]}"
    significant = p is not None and not math.isnan(p) and p < SIGNIFICANCE
    if skill > MATERIAL_SKILL and significant:
        level = "aporta valor"
        head = (
            f"En el periodo de prueba ({period}), la red {arch_label} acertó más que la regla "
            f"«el precio no cambia»: redujo el error medio un {_pct(skill, False)} y la "
            "diferencia es estadísticamente significativa."
        )
    elif skill > 0:
        level = "mejora no concluyente"
        head = (
            f"En el periodo de prueba ({period}), la red {arch_label} redujo el error medio un "
            f"{_pct(skill, False)} frente a suponer que el precio no cambia, pero la mejora "
            "no es estadísticamente concluyente: tome la cifra central como orientativa."
        )
    else:
        level = "no mejora al ingenuo"
        head = (
            f"En el periodo de prueba ({period}), la red {arch_label} no mejoró a la regla «el "
            f"precio no cambia» (su error medio fue un {_pct(-skill, False)} mayor). A este "
            "horizonte la cifra central no aporta más que el precio actual: tómela solo como "
            "referencia y fíjese sobre todo en el rango, que indica cuánto podría moverse el "
            "precio."
        )
    body = (
        f" Error medio en ese periodo: {_money(metrics['mae'])}/barril "
        f"(ingenuo: {_money(naive['mae'])}). El intervalo del 80 % contuvo el precio real en "
        f"el {_pct(metrics['coverage80'], False)} de los casos."
    )
    if skill_by_year:
        worst = min(skill_by_year, key=skill_by_year.get)
        best = max(skill_by_year, key=skill_by_year.get)
        if worst != best:
            body += (
                f" El resultado varía mucho según el año: el mejor fue {best} "
                f"({_pct(skill_by_year[best])} de mejora) y el peor {worst} "
                f"({_pct(skill_by_year[worst])})."
            )
    return head + body, level


def risks_for(h: int, recent_vol_pct: float | None) -> list[str]:
    risks = [
        "Acontecimientos geopolíticos (conflictos en Oriente Medio, el estrecho de Ormuz, "
        "sanciones) que cambian de golpe la oferta disponible.",
        "Decisiones de producción de la OPEP+.",
        "Un cambio brusco del ciclo económico (recesión, frenazo de China) que altere la demanda.",
    ]
    if h >= 63:
        risks.append(
            "A varios meses vista pesan más los cambios estructurales (política "
            "monetaria, tipo de cambio del dólar, inventarios, transición energética), "
            "que el modelo solo ve a través del pasado reciente."
        )
    if recent_vol_pct is not None and recent_vol_pct >= 0.9:
        risks.append(
            f"La volatilidad del último mes está en el percentil "
            f"{round(recent_vol_pct * 100)} de la serie histórica: en periodos así los "
            "errores suelen ser mayores que la media."
        )
    risks.append(
        "Los precios spot del Brent se publican con unos días de retraso: la previsión "
        "parte del último dato oficial disponible, no del precio de hoy."
    )
    return risks


def explain(
    fc: HorizonForecast, hspec: dict, arch_label: str, recent_vol_pct: float | None = None
) -> Explanation:
    lo, hi = fc.intervals["80"]
    direction = ("subida" if fc.change > 0 else "bajada") if abs(fc.change_pct) >= 0.005 else None
    move = (
        f"una {direction} del {_pct(abs(fc.change_pct), False)}"
        if direction
        else "un precio prácticamente sin cambios"
    )
    summary = (
        f"A {HORIZON_NAMES.get(fc.horizon, f'{fc.horizon} días hábiles')}, hacia el "
        f"{_date(fc.target_date)}, el modelo sitúa el Brent en torno a "
        f"{_money(fc.median)}/barril: {move} respecto al último precio publicado "
        f"({_money(fc.last_price)} el {_date(fc.origin_date)}). Con una probabilidad aproximada "
        f"del 80 %, el precio estaría entre {_money(lo)} y {_money(hi)}."
    )
    reliability, level = reliability_text(
        fc.horizon,
        hspec["test_metrics"],
        hspec["naive_metrics"],
        {int(k): v for k, v in hspec.get("skill_by_year", {}).items()},
        arch_label,
    )
    return Explanation(summary, reliability, level, risks_for(fc.horizon, recent_vol_pct))
