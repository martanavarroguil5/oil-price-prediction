"""Versión estática de la web para GitHub Pages.

Escribe en una carpeta lo que el servidor local calcula al vuelo: el estado de
``/api/state``, las series de validación y las previsiones de la prueba histórica. La web
detecta el modo estático por una etiqueta ``<meta>`` y lee estos ficheros en lugar de la API.
Actualizar datos y reentrenar solo funcionan con el servidor local.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from oilcast.config import Config
from oilcast.data.dataset import load_dataset
from oilcast.paths import Paths
from oilcast.replay import replay_frame
from oilcast.web import FIGURE_NAMES, snapshot, validation

ARCHS = ("rnn", "lstm", "gru")
HORIZONS = (5, 63, 252)
STATIC_META = '<meta name="estrato-mode" content="static" />'


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def _replay_cases(paths: Paths, arch: str, horizon: int) -> dict:
    """Todas las previsiones de test de una red y un plazo, con su resultado."""
    frame = replay_frame(paths, arch, horizon)
    price = frame["price"]
    rows = zip(
        frame.index.strftime("%Y-%m-%d"),
        frame["target_date"].dt.strftime("%Y-%m-%d"),
        frame["refit_date"].dt.strftime("%Y-%m-%d"),
        price.round(4),
        (price * np.exp(frame["q50"])).round(4),
        (price * np.exp(frame["lo80"])).round(4),
        (price * np.exp(frame["hi80"])).round(4),
        (price * np.exp(frame["y"])).round(4),
        strict=True,
    )
    return {
        "columns": ["origin", "target", "refit", "price", "median", "lo", "hi", "actual"],
        "rows": [
            [o, t, r, float(p), float(m), float(lo), float(hi), None if np.isnan(a) else float(a)]
            for o, t, r, p, m, lo, hi, a in rows
        ],
    }


def export_static(cfg: Config, paths: Paths, out: Path) -> Path:
    """Genera la web estática en ``out`` (se vacía antes) y devuelve la ruta."""
    static = paths.root / "app" / "static"
    if out.resolve() in {paths.root.resolve(), static.resolve()}:
        raise ValueError("La carpeta de salida no puede ser el proyecto ni app/static")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(static, out)
    index = out / "index.html"
    html = index.read_text(encoding="utf-8")
    index.write_text(html.replace("<head>", "<head>\n    " + STATIC_META, 1), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # sirve carpetas que empiezan por _

    for name in FIGURE_NAMES:
        src = paths.figures / name
        if src.exists():
            (out / "figures").mkdir(exist_ok=True)
            shutil.copy2(src, out / "figures" / name)
    shutil.copy2(paths.reports / "comparativa.md", out / "comparativa.md")

    api = out / "api"
    _write(api / "state.json", snapshot(cfg, paths))
    for arch in (*ARCHS, "naive"):
        for h in HORIZONS:
            _write(api / "validation" / f"{arch}_{h}.json", validation(paths, arch, h))
    for arch in ARCHS:
        for h in HORIZONS:
            _write(api / "replay" / f"{arch}_{h}.json", _replay_cases(paths, arch, h))
    brent = load_dataset(paths)["brent"].dropna()
    _write(
        api / "brent.json",
        {"dates": brent.index.strftime("%Y-%m-%d").tolist(), "prices": brent.round(4).tolist()},
    )
    return out
