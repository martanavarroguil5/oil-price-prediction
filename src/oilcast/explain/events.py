"""Economic-cycle view: curated catalogue of major Brent moves + automatic turning points.

Each catalogue entry gives a search window; the actual extreme (max for peaks, min for
troughs) is located in the data, so the marker always sits on the real price. Figures in
the texts were checked against EIA/IEA publications; where the cause of a move is not
clearly established the wording stays descriptive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass(frozen=True)
class CycleEvent:
    start: str
    end: str
    kind: str  # "peak" | "trough"
    title: str
    phase: str  # relation with the world economic cycle
    text: str


CATALOGUE: list[CycleEvent] = [
    CycleEvent(
        "2008-01-01",
        "2008-09-01",
        "peak",
        "Máximo histórico de 2008",
        "Final de una expansión mundial (superciclo de materias primas)",
        "El Brent marcó su máximo histórico en julio de 2008 tras años de fuerte crecimiento "
        "de la demanda de las economías emergentes (sobre todo China), una oferta poco "
        "flexible, un dólar débil y fuertes entradas de inversión financiera en materias "
        "primas.",
    ),
    CycleEvent(
        "2008-09-01",
        "2009-06-30",
        "trough",
        "Desplome por la crisis financiera",
        "Recesión mundial (shock de demanda)",
        "La quiebra de Lehman Brothers (septiembre de 2008) desencadenó la crisis financiera "
        "global. La recesión hundió la demanda de crudo y el precio cayó más de un 75 % en "
        "menos de seis meses, hasta que los recortes de la OPEP y la recuperación de 2009 lo "
        "estabilizaron.",
    ),
    CycleEvent(
        "2011-01-01",
        "2012-12-31",
        "peak",
        "Primavera Árabe y sanciones a Irán",
        "Recuperación económica con riesgo geopolítico de oferta",
        "La guerra de Libia (2011) retiró su producción del mercado y las sanciones al crudo "
        "iraní (2012) mantuvieron el Brent por encima de 100 $ durante 2011-2014, en plena "
        "recuperación tras la crisis.",
    ),
    CycleEvent(
        "2014-01-01",
        "2014-12-31",
        "peak",
        "Inicio de la caída de 2014-2016",
        "Fin del ciclo de precios altos (exceso de oferta)",
        "A mediados de 2014 empezó una caída prolongada: el auge del petróleo de esquisto "
        "(shale) en EE. UU. elevó la oferta mientras la demanda se moderaba y el dólar se "
        "fortalecía.",
    ),
    CycleEvent(
        "2015-06-01",
        "2016-06-30",
        "trough",
        "Mínimo de 2016 por exceso de oferta",
        "Exceso de oferta y desaceleración de emergentes",
        "En noviembre de 2014 la OPEP decidió no recortar producción para defender su cuota "
        "frente al shale, y el mercado acumuló excedentes. El Brent tocó fondo en enero de "
        "2016; a finales de ese año la OPEP y otros productores (OPEP+) acordaron recortes.",
    ),
    CycleEvent(
        "2020-01-01",
        "2020-12-31",
        "trough",
        "Desplome por la pandemia",
        "Recesión por la COVID-19 (shock de demanda extremo)",
        "Los confinamientos hundieron la demanda de combustibles, coincidiendo con la guerra "
        "de precios entre Arabia Saudí y Rusia (marzo de 2020). Con los depósitos casi "
        "llenos, el WTI cotizó en negativo el 20 de abril de 2020 y el Brent cayó a su mínimo "
        "del siglo al día siguiente. En abril la OPEP+ pactó un recorte récord de producción.",
    ),
    CycleEvent(
        "2022-01-01",
        "2022-12-31",
        "peak",
        "Invasión rusa de Ucrania",
        "Recuperación post-pandemia + shock geopolítico de oferta",
        "La invasión de Ucrania (24 de febrero de 2022) y las sanciones a Rusia, gran "
        "exportador, dispararon el precio en un mercado ya tensionado por la recuperación de "
        "la demanda tras la pandemia. Después cedió con las subidas de tipos y el temor a una "
        "recesión.",
    ),
    CycleEvent(
        "2025-06-01",
        "2026-02-27",
        "trough",
        "Expectativas de exceso de oferta (2025)",
        "Crecimiento moderado y holgura de oferta",
        "Antes de la guerra de 2026 el precio rondaba los 60-70 $: la OPEP+ había ido "
        "aumentando su producción durante 2025 y las agencias anticipaban un excedente de "
        "oferta para 2026.",
    ),
    CycleEvent(
        "2026-02-28",
        "2026-05-31",
        "peak",
        "Guerra con Irán y cierre de Ormuz",
        "Shock geopolítico de oferta",
        "Tras los ataques de EE. UU. e Israel contra Irán (28 de febrero de 2026) el tráfico "
        "por el estrecho de Ormuz quedó prácticamente paralizado. El Brent spot alcanzó unos "
        "138 $ el 7 de abril, su nivel más alto desde 2008, con el crudo físico cotizando muy "
        "por encima de los futuros.",
    ),
    CycleEvent(
        "2026-05-01",
        "2026-08-15",
        "trough",
        "Retroceso de mediados de 2026",
        "Alivio parcial del shock de oferta",
        "El precio cayó con fuerza entre mayo y julio. Según la EIA, la bajada de junio "
        "coincidió con un memorando de entendimiento entre EE. UU. e Irán (17 de junio); la "
        "media de junio quedó en torno a 85 $.",
    ),
    CycleEvent(
        "2026-08-15",
        "2026-11-30",
        "peak",
        "Segundo repunte (septiembre-octubre 2026)",
        "Shock geopolítico de oferta",
        "La intensificación de los ataques contra infraestructuras petroleras de Oriente "
        "Medio volvió a disparar el precio: según la EIA, el Brent promedió 114 $ en "
        "septiembre, y en octubre superó de nuevo los 130 $.",
    ),
]


def locate_events(price: pd.Series, catalogue: list[CycleEvent] = CATALOGUE) -> pd.DataFrame:
    """Place each catalogue event on the actual extreme of its window (events without data
    in the window are skipped)."""
    rows = []
    for ev in catalogue:
        s = price.loc[ev.start : ev.end].dropna()
        if s.empty:
            continue
        date = s.idxmax() if ev.kind == "peak" else s.idxmin()
        rows.append(
            {
                "date": date,
                "price": float(s.loc[date]),
                "kind": ev.kind,
                "title": ev.title,
                "phase": ev.phase,
                "text": ev.text,
            }
        )
    return pd.DataFrame(rows)


def detect_turning_points(
    price: pd.Series, prominence: float = 0.35, min_distance_days: int = 120
) -> pd.DataFrame:
    """Major peaks/troughs by prominence on log prices (0.35 ~ a 40 % swing)."""
    lp = np.log(price.dropna())
    rows = []
    for sign, kind in ((1, "peak"), (-1, "trough")):
        idx, props = find_peaks(
            sign * lp.to_numpy(), prominence=prominence, distance=min_distance_days
        )
        for i, prom in zip(idx, props["prominences"], strict=True):
            rows.append(
                {
                    "date": lp.index[i],
                    "price": float(np.exp(lp.iloc[i])),
                    "kind": kind,
                    "prominence": float(prom),
                }
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def cycle_markers(price: pd.Series, tolerance_days: int = 90) -> pd.DataFrame:
    """Catalogue events plus detected turning points not already explained by one."""
    events = locate_events(price)
    events["source"] = "catalogue"
    detected = detect_turning_points(price)
    extra = []
    for r in detected.itertuples():
        near = (events["kind"] == r.kind) & (
            (events["date"] - r.date).abs() <= pd.Timedelta(days=tolerance_days)
        )
        if not near.any():
            label = "Pico relevante" if r.kind == "peak" else "Caída relevante"
            extra.append(
                {
                    "date": r.date,
                    "price": r.price,
                    "kind": r.kind,
                    "title": label,
                    "phase": "Detectado automáticamente",
                    "text": "Giro importante del precio detectado automáticamente "
                    "(prominencia > 35 % en escala logarítmica).",
                    "source": "detected",
                }
            )
    return pd.concat([events, pd.DataFrame(extra)], ignore_index=True).sort_values("date")
