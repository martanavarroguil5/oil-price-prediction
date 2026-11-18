"""Command-line interface: ``oilcast <command>`` (run ``oilcast --help``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from oilcast.config import Config, load_config
from oilcast.paths import Paths, get_paths
from oilcast.utils import setup_logging

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Previsión del Brent con RNN, LSTM y GRU."
)
log = logging.getLogger("oilcast")
_state: dict = {}

ArchOpt = Annotated[list[str] | None, typer.Option("--arch", "-a", help="rnn | lstm | gru")]
HorizonOpt = Annotated[list[int] | None, typer.Option("--horizon", "-h", help="5 | 63 | 252")]


@app.callback()
def main(config: Annotated[Path | None, typer.Option("--config", "-c")] = None) -> None:
    setup_logging()
    cfg = load_config(config)
    paths = get_paths(cfg)
    paths.ensure()
    _state.update(cfg=cfg, paths=paths)


def _ctx() -> tuple[Config, Paths]:
    return _state["cfg"], _state["paths"]


def _grid(arch: list[str] | None, horizon: list[int] | None) -> list[tuple[str, int]]:
    cfg, _ = _ctx()
    return [
        (a, h)
        for a in (arch or cfg["architectures"])
        for h in (horizon or cfg["targets"]["horizons"])
    ]


@app.command()
def download() -> None:
    """Descarga las series en bruto (FRED, Yahoo, OPEP) a data/raw."""
    from oilcast.data.download import download_all

    download_all(*_ctx())


@app.command()
def build() -> None:
    """Construye data/processed/brent_dataset.csv sobre el calendario del Brent."""
    from oilcast.data.dataset import build_dataset

    build_dataset(*_ctx())


@app.command()
def tune(arch: ArchOpt = None, horizon: HorizonOpt = None) -> None:
    """Búsqueda de hiperparámetros (Optuna) con CV walk-forward purgada."""
    from oilcast.training.tune import run_study

    for a, h in _grid(arch, horizon):
        run_study(*_ctx(), a, h)


@app.command()
def evaluate(arch: ArchOpt = None, horizon: HorizonOpt = None) -> None:
    """Evaluación walk-forward en test con reentrenos anuales y varias semillas."""
    from oilcast.training.evaluate import run_evaluation

    for a, h in _grid(arch, horizon):
        run_evaluation(*_ctx(), a, h)


@app.command()
def timing() -> None:
    """Mide el tiempo de entrenamiento de cada red (secuencial, sin otros procesos)."""
    from oilcast.training.benchmark import run_timing

    run_timing(*_ctx())


@app.command()
def baselines() -> None:
    """Compara modelos sencillos por horizonte (usar -c configs/baselines.yaml)."""
    from oilcast.evaluation.baselines import run_baselines

    cfg, paths = _ctx()
    if "baselines" not in cfg:
        raise typer.BadParameter("Usa -c configs/baselines.yaml para esta comparación")
    run_baselines(cfg, paths)


@app.command("returns")
def returns_report() -> None:
    """Exporta errores en rentabilidad simple (puntos porcentuales), sin reentrenar."""
    import pandas as pd

    from oilcast.evaluation.returns import return_comparison

    _, paths = _ctx()
    output = paths.reports / "rentabilidad.csv"
    rows = return_comparison(paths)
    if not rows:
        raise typer.BadParameter("No hay predicciones guardadas para evaluar")
    pd.DataFrame(rows).to_csv(output, index=False)
    typer.echo(f"Comparación en rentabilidad: {output}")


@app.command()
def summarize() -> None:
    """Calcula métricas, tabla comparativa y elige la arquitectura de producción."""
    from oilcast.evaluation.summary import summarize as run

    run(*_ctx())


@app.command("train-final")
def train_final(
    arch: Annotated[str | None, typer.Option(help="Forzar arquitectura")] = None,
    from_metadata: Annotated[
        bool,
        typer.Option(
            help="Reutilizar la especificación guardada en models/ (reentrenado periódico)"
        ),
    ] = False,
) -> None:
    """Entrena el modelo de producción con todos los datos y lo guarda en models/."""
    from oilcast.training.final import spec_from_metadata, spec_from_results
    from oilcast.training.final import train_final as run

    cfg, paths = _ctx()
    has_results = (paths.results / "selection.json").exists()
    spec = (
        spec_from_metadata(paths)
        if from_metadata or not has_results
        else spec_from_results(cfg, paths, arch)
    )
    run(cfg, paths, spec)


@app.command()
def serve(port: Annotated[int, typer.Option(min=1, max=65535)] = 8501) -> None:
    """Abre el servidor local del observatorio web."""
    from oilcast.web import serve as run

    run(*_ctx(), port)


@app.command()
def forecast() -> None:
    """Muestra la previsión actual para los tres horizontes."""
    from oilcast.data.dataset import load_dataset
    from oilcast.inference.forecaster import Forecaster

    cfg, paths = _ctx()
    fc = Forecaster(paths.models).forecast(load_dataset(paths), cfg)
    for h, f in fc.items():
        lo, hi = f.intervals["80"]
        typer.echo(
            f"h={h:>3}  {f.origin_date.date()} -> ~{f.target_date.date()}  "
            f"último {f.last_price:7.2f}  previsto {f.median:7.2f} "
            f"({f.change_pct:+.1%})  IC80 [{lo:.2f}, {hi:.2f}]"
        )


@app.command()
def report() -> None:
    """Genera reports/comparativa.md y las figuras a partir de los resultados."""
    from oilcast.evaluation.report import build_report

    build_report(*_ctx())


@app.command()
def diagnostics() -> None:
    """Genera velas de futuros y violines de errores con las predicciones guardadas."""
    from oilcast.evaluation.diagnostics import build_diagnostics

    _, paths = _ctx()
    build_diagnostics(paths)
    typer.echo(f"Velas y violines: {paths.figures}")


@app.command("improve-intervals")
def improve_intervals() -> None:
    """Evalúa mejoras de intervalos (-c configs/intervals.yaml), sin reentrenar redes."""
    from oilcast.evaluation.interval_study import run_interval_study

    cfg, paths = _ctx()
    if "interval_study" not in cfg:
        raise typer.BadParameter("Usa -c configs/intervals.yaml")
    decision = run_interval_study(cfg, paths)
    typer.echo(str(decision["selected"]))


@app.command("download-extra")
def download_extra() -> None:
    """Experimento ampliado: descarga las series extra (geopolítica, incertidumbre, macro)."""
    from oilcast.data.extra import download_extra as run

    run(*_ctx())


@app.command("build-extra")
def build_extra() -> None:
    """Experimento ampliado: alinea las series extra al calendario del dataset principal."""
    from oilcast.data.extra import build_extra as run

    run(_ctx()[1])


@app.command("all")
def run_all(skip_download: bool = True) -> None:
    """Pipeline completo en secuencia."""
    if not skip_download:
        download()
    build()
    tune()
    evaluate()
    timing()
    summarize()
    train_final()
    report()


if __name__ == "__main__":
    app()
