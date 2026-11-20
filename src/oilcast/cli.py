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


@app.command("export-static")
def export_static(
    out: Annotated[Path, typer.Option("--out", help="Carpeta de salida")] = Path("site"),
) -> None:
    """Genera la versión estática de la web (GitHub Pages) en ``out``."""
    from oilcast.static_site import export_static as run

    typer.echo(run(*_ctx(), out))


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


@app.command("download-products")
def download_products() -> None:
    """Ronda 3: descarga gasolina y gasóleo de calefacción (EIA vía FRED)."""
    from oilcast.data.extra import download_products as run

    for name, msg in run(*_ctx()).items():
        typer.echo(f"{name}: {msg}")


@app.command("multi-tune")
def multi_tune(arch: ArchOpt = None) -> None:
    """Ronda 3: búsqueda de hiperparámetros de la red multihorizonte (-c configs/ronda3*.yaml)."""
    from oilcast.training.multitask import run_study

    cfg, paths = _ctx()
    for a in arch or cfg["architectures"]:
        run_study(cfg, paths, a)


@app.command("multi-evaluate")
def multi_evaluate(arch: ArchOpt = None) -> None:
    """Ronda 3: predicciones walk-forward de la red multihorizonte."""
    from oilcast.training.multitask import run_evaluation

    cfg, paths = _ctx()
    for a in arch or cfg["architectures"]:
        run_evaluation(cfg, paths, a)


@app.command("extend-published")
def extend_published_cmd(
    arch: ArchOpt = None,
    horizon: HorizonOpt = None,
    source: Annotated[str, typer.Option(help="Resultados publicados")] = "reports/results",
) -> None:
    """Ronda 3: añade los reentrenos de 2008-2012 a las predicciones publicadas (W1)."""
    from oilcast.training.multitask import extend_published

    for a, h in _grid(arch, horizon):
        extend_published(*_ctx(), a, h, source)


@app.command("ronda3-decide")
def ronda3_decide() -> None:
    """Ronda 3: aplica la regla con datos de desarrollo y guarda la decisión (sin test)."""
    from oilcast.evaluation.ronda3 import decide

    d = decide(_ctx()[0])
    typer.echo(f"mejor: {d['best']}\nadoptar: {d['adopt']} -> {d['chosen']}")


@app.command("ronda3-test")
def ronda3_test() -> None:
    """Ronda 3: calcula una sola vez el test de todas las variantes (tras ronda3-decide)."""
    from oilcast.evaluation.ronda3 import test_report

    test_report(*_ctx())
    typer.echo("reports/experiments/ronda3/test.csv")


@app.command("ronda4-decide")
def ronda4_decide() -> None:
    """Ronda 4: aplica la regla con datos de desarrollo y guarda la decisión (sin test)."""
    from oilcast.evaluation.ronda4 import decide4

    d = decide4(_ctx()[0])
    typer.echo(f"mejor: {d['best']}\nadoptar: {d['adopt']} -> {d['chosen']}")


@app.command("ronda4-test")
def ronda4_test() -> None:
    """Ronda 4: calcula una sola vez el test (tras ronda4-decide) y la referencia lineal."""
    from oilcast.evaluation.ronda4 import test4

    test4(*_ctx())
    typer.echo("reports/experiments/ronda4/test.csv")


@app.command("ronda5-decide")
def ronda5_decide() -> None:
    """Ronda 5: puntúa las candidatas en desarrollo y guarda la decisión (sin test)."""
    from oilcast.evaluation.ronda5 import decide5

    d = decide5()
    for h, c in d["choice"].items():
        typer.echo(f"h={h}: merecen la pena {c['worth_it']} -> {c['chosen']}")


@app.command("ronda5-test")
def ronda5_test() -> None:
    """Ronda 5: calcula una sola vez el test de todas las candidatas (tras ronda5-decide)."""
    from oilcast.evaluation.ronda5 import test5

    test5()
    typer.echo("reports/experiments/ronda5/test.csv")


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
