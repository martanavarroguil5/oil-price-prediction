"""Local web application for the Brent observatory. Serve the browser and model on one origin."""

from __future__ import annotations

import json
import logging
import mimetypes
import shutil
import threading
from dataclasses import asdict, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import torch

from oilcast.config import Config
from oilcast.data.dataset import build_dataset, load_dataset
from oilcast.data.download import download_all
from oilcast.evaluation.returns import return_comparison
from oilcast.explain.events import cycle_markers
from oilcast.explain.narrative import explain
from oilcast.features.engineering import compute_features
from oilcast.inference.forecaster import Forecaster
from oilcast.inference.interval_preview import interval_preview
from oilcast.paths import Paths
from oilcast.replay import replay_case, replay_dates
from oilcast.training.final import spec_from_metadata, train_final

log = logging.getLogger(__name__)

# Figuras de reports/figures que la web puede mostrar.
FIGURE_NAMES = frozenset(
    {"mejora_vs_ingenuo.png", "mejora_por_anio.png", "predicciones_test.png"}
    | {"brent_velas.png", "violines_precio.png", "violines_retorno.png"}
    | {f"dispersion_{arch}.png" for arch in ("rnn", "lstm", "gru")}
    | {f"violines_{unit}_h{h}.png" for unit in ("precio", "retorno") for h in (5, 63, 252)}
)


def records(path: Path) -> list[dict]:
    return json.loads(pd.read_csv(path).to_json(orient="records")) if path.exists() else []


def snapshot(cfg: Config, paths: Paths) -> dict:
    df = load_dataset(paths)
    model = Forecaster(paths.models)
    meta = model.meta
    vol = compute_features(df, cfg)["rv_21"].dropna()
    vol_pct = float((vol <= vol.iloc[-1]).mean()) if len(vol) else None
    forecasts = {}
    current_forecasts = model.forecast(df, cfg)
    for h, fc in current_forecasts.items():
        explanation = explain(fc, meta["spec"]["horizons"][str(h)], meta["arch_label"], vol_pct)
        forecasts[h] = {
            "price": fc.median,
            "date": fc.target_date.date().isoformat(),
            "change": fc.change,
            "changePct": fc.change_pct,
            "intervals": fc.intervals,
            "explanation": asdict(explanation),
        }
    events = cycle_markers(df["brent"])
    events["date"] = events["date"].dt.strftime("%Y-%m-%d")
    history = pd.DataFrame(
        {
            "date": df.index.strftime("%Y-%m-%d"),
            "brent": df["brent"],
            "ma50": df["brent"].rolling(50).mean(),
            "ma200": df["brent"].rolling(200).mean(),
        }
    )
    return {
        "date": history.iloc[-1]["date"],
        "history": json.loads(history.to_json(orient="values")),
        "forecasts": forecasts,
        "events": json.loads(events.to_json(orient="records")),
        "model": {
            k: meta[k]
            for k in (
                "arch",
                "arch_label",
                "window",
                "trained_at",
                "last_data_date",
                "quantiles",
                "seeds",
            )
        },
        # variables de entrada por plazo (a 252 sesiones solo oro/crudo, ronda 4)
        "features": {
            hs: len(info.get("feature_names", meta["feature_names"]))
            for hs, info in meta["fit"].items()
        },
        "rule": meta["spec"]["selection_rule"],
        # variantes que ganan a la red de la app en algún plazo del test y por qué no se usan
        "alternatives": (cfg.get("selection") or {}).get("alternatives", []),
        "alternativesNote": (cfg.get("selection") or {}).get("alternatives_note", ""),
        "testStart": cfg["splits"]["test_start"],
        "comparison": records(paths.results / "comparison.csv"),
        "returnMetrics": return_comparison(paths),
        "intervalStudy": interval_preview(paths, current_forecasts, meta["arch"]),
        "diagnostics": (
            json.loads((paths.figures / "diagnostics.json").read_text(encoding="utf-8"))
            if (paths.figures / "diagnostics.json").exists()
            else None
        ),
        "years": records(paths.results / "metrics_by_year.csv"),
        "architecture": records(paths.results / "timing.csv"),
    }


def validation(paths: Paths, arch: str, horizon: int) -> list:
    filename = f"naive_h{horizon}.csv" if arch == "naive" else f"{arch}_h{horizon}_ensemble.csv"
    df = pd.read_csv(paths.results / "predictions" / filename)
    t = df.loc[df.phase == "test"]
    predicted = t.price * np.exp(t.q50)
    actual = t.price * np.exp(t.y)
    return json.loads(
        pd.DataFrame(
            {
                "date": t.target_date,
                "actual": actual,
                "model": predicted,
                "naive": t.price,
            }
        ).to_json(orient="values")
    )


def stage_refresh(cfg: Config, paths: Paths, stage: Path) -> tuple[list[tuple[Path, Path]], str]:
    """Descarga los datos en ``stage`` y devuelve los pares (nuevo, destino) a publicar.

    Falla sin tocar la copia local si el Brent no responde o la serie nueva termina antes.
    """
    staged = replace(paths, raw=stage / "raw", processed=stage / "processed")
    shutil.copytree(paths.raw, staged.raw)
    sources = download_all(cfg, staged)
    if not sources.get("brent", "").startswith("ok"):
        raise ValueError("La fuente del Brent no ha respondido. Se conserva la serie local.")
    frame = build_dataset(cfg, staged)
    if frame.index[-1] < load_dataset(paths).index[-1]:
        raise ValueError("La serie descargada termina antes que la copia local.")
    pairs = [(p, paths.raw / p.name) for p in staged.raw.glob("*.csv")]
    pairs += [(staged.cleaning_log, paths.cleaning_log), (staged.dataset_csv, paths.dataset_csv)]
    ok = sum(s.startswith("ok") for s in sources.values())
    return pairs, f"Serie hasta el {frame.index[-1]:%d/%m/%Y}. Fuentes: {ok}/{len(sources)}."


class Observatory:
    def __init__(self, cfg: Config, paths: Paths):
        self.cfg, self.paths = cfg, paths
        self.lock = threading.RLock()
        self.cache_key = None
        self.cache = None
        self.job = {"status": "idle", "message": "", "progress": 0}

    def state(self):
        with self.lock:
            files = [self.paths.dataset_csv, self.paths.models / "metadata.json"]
            files.append(self.paths.figures / "diagnostics.json")
            files += [
                self.paths.reports / "experiments/intervals" / filename
                for filename in ("app.json", "decision.json")
            ]
            files += [
                self.paths.results / name
                for name in ("comparison.csv", "timing.csv", "metrics_by_year.csv")
            ]
            files += [
                self.paths.results / "predictions" / f"{arch}_h{h}_ensemble.csv"
                for arch in ("rnn", "lstm", "gru")
                for h in (5, 63, 252)
            ]
            key = tuple(f.stat().st_mtime_ns if f.exists() else None for f in files)
            if key != self.cache_key:
                self.cache = snapshot(self.cfg, self.paths)
                self.cache_key = key
            return self.cache

    def start(self, action: str, mode: str):
        with self.lock:
            if self.job["status"] == "running":
                return False
            self.job = {"status": "running", "message": "Preparando…", "progress": 0}
            threading.Thread(target=self._run, args=(action, mode), daemon=True).start()
            return True

    def _progress(self, value, message):
        with self.lock:
            self.job.update(message=message, progress=value)

    def _run(self, action, mode):
        paths = self.paths
        try:
            with TemporaryDirectory(dir=paths.root) as tmp:
                stage = Path(tmp)
                if action == "refresh":
                    self._progress(0.15, "Descargando las últimas observaciones…")
                    pairs, message = stage_refresh(self.cfg, paths, stage)
                else:
                    staged = replace(paths, models=stage / "models")
                    seeds = [0] if mode == "fast" else self.cfg["final"]["seeds"]
                    train_final(
                        self.cfg,
                        staged,
                        spec_from_metadata(paths),
                        seeds=seeds,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        progress=self._progress,
                    )
                    Forecaster(staged.models).forecast(load_dataset(paths), self.cfg)
                    pairs = [(p, paths.models / p.name) for p in staged.models.glob("*.pt")]
                    pairs.append((staged.models / "metadata.json", paths.models / "metadata.json"))
                    message = "Modelo actualizado. Las nuevas previsiones están disponibles."
                with self.lock:
                    # Restore the previous set if a file fails to publish mid-operation.
                    backups = []
                    try:
                        for index, (src, dst) in enumerate(pairs):
                            previous = stage / f"backup-{index}" if dst.exists() else None
                            if previous:
                                shutil.copy2(dst, previous)
                            src.replace(dst)
                            backups.append((dst, previous))
                    except OSError:
                        for dst, previous in reversed(backups):
                            if previous:
                                previous.replace(dst)
                            else:
                                dst.unlink(missing_ok=True)
                        raise
                    self.cache_key = None
                    self.job.update(status="done", progress=1, message=message)
        except Exception as exc:
            log.exception("Operation %s failed", action)
            with self.lock:
                self.job.update(status="error", message=str(exc), progress=0)


def handler(observatory: Observatory):
    static = observatory.paths.root / "app" / "static"
    figure_names = FIGURE_NAMES

    class Handler(BaseHTTPRequestHandler):
        def send_content(self, status, body, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_json(self, value, status=200):
            self.send_content(
                status, json.dumps(value, allow_nan=False).encode(), "application/json"
            )

        def do_HEAD(self):
            return self.do_GET()

        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == "/api/state":
                    return self.send_json(observatory.state())
                if route == "/api/job":
                    with observatory.lock:
                        return self.send_json(observatory.job.copy())
                if route.startswith("/api/replay/"):
                    parts = route.strip("/").split("/")
                    if (
                        len(parts) not in {4, 5, 6}
                        or parts[2] not in {"rnn", "lstm", "gru"}
                        or parts[3] not in {"5", "63", "252"}
                        or (len(parts) == 6 and parts[5] != "reveal")
                    ):
                        return self.send_error(404)
                    try:
                        if len(parts) == 4:
                            return self.send_json(
                                replay_dates(observatory.paths, parts[2], int(parts[3]))
                            )
                        return self.send_json(
                            replay_case(
                                observatory.paths,
                                parts[2],
                                int(parts[3]),
                                parts[4],
                                reveal=len(parts) == 6,
                            )
                        )
                    except ValueError as exc:
                        return self.send_json({"error": str(exc)}, 404)
                if route.startswith("/api/validation/"):
                    parts = route.split("/")
                    if (
                        len(parts) != 5
                        or parts[3] not in {"rnn", "lstm", "gru", "naive"}
                        or parts[4] not in {"5", "63", "252"}
                    ):
                        return self.send_error(404)
                    return self.send_json(validation(observatory.paths, parts[3], int(parts[4])))
                if route == "/comparativa.md":
                    path = observatory.paths.reports / "comparativa.md"
                elif route.startswith("/figures/") and route[9:] in figure_names:
                    path = observatory.paths.figures / route[9:]
                else:
                    path = (static / (route.lstrip("/") or "index.html")).resolve()
                    if not path.is_relative_to(static.resolve()):
                        return self.send_error(404)
                if not path.is_file():
                    return self.send_error(404)
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                return self.send_content(200, path.read_bytes(), mime)
            except (BrokenPipeError, ConnectionResetError):
                return None
            except FileNotFoundError:
                return self.send_json({"error": "Faltan datos o resultados en el proyecto."}, 404)
            except Exception:
                log.exception("Cannot serve %s", route)
                return self.send_json(
                    {"error": "No se han podido cargar los datos del observatorio."}, 500
                )

        def do_POST(self):
            action = self.path.removeprefix("/api/")
            if self.path not in {"/api/retrain", "/api/refresh"}:
                return self.send_error(404)
            hosts = {f"{host}:{self.server.server_port}" for host in ("localhost", "127.0.0.1")}
            if self.headers.get("Host") not in hosts:
                return self.send_error(HTTPStatus.FORBIDDEN)
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send_error(HTTPStatus.FORBIDDEN)
            if self.headers.get_content_type() != "application/json":
                return self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 256:
                    raise ValueError
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or data.get("mode", "fast") not in ("fast", "full"):
                    raise ValueError
            except (ValueError, TypeError):
                return self.send_error(400)
            if not observatory.start(action, data.get("mode", "fast")):
                return self.send_json({"error": "Ya hay una operación en curso."}, 409)
            return self.send_json({"status": "running"}, 202)

    return Handler


def serve(cfg: Config, paths: Paths, port: int = 8501):
    torch.set_num_threads(cfg["training"]["num_threads"])
    with ThreadingHTTPServer(("127.0.0.1", port), handler(Observatory(cfg, paths))) as server:
        print(f"Estrato · http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()
