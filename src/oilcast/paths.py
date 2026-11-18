"""Project paths. Output directories can be redirected from the config (e.g. smoke runs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    results: Path
    models: Path
    figures: Path
    reports: Path

    @property
    def dataset_csv(self) -> Path:
        return self.processed / "brent_dataset.csv"

    @property
    def extra_csv(self) -> Path:
        """Extra series of the extended experiment, on the same Brent calendar."""
        return self.processed / "extra_dataset.csv"

    @property
    def cleaning_log(self) -> Path:
        return self.processed / "cleaning_log.json"

    def tuning_dir(self, arch: str, horizon: int) -> Path:
        return self.results / "tuning" / f"{arch}_h{horizon}"

    def predictions_csv(self, arch: str, horizon: int) -> Path:
        return self.results / "predictions" / f"{arch}_h{horizon}.csv"

    def ensure(self) -> None:
        for p in (self.raw, self.processed, self.results, self.models, self.figures):
            p.mkdir(parents=True, exist_ok=True)


def get_paths(cfg: dict[str, Any] | None = None, root: Path = PROJECT_ROOT) -> Paths:
    out = (cfg or {}).get("paths", {})
    return Paths(
        root=root,
        raw=root / out.get("raw_dir", "data/raw"),
        processed=root / out.get("processed_dir", "data/processed"),
        results=root / out.get("results_dir", "reports/results"),
        models=root / out.get("models_dir", "models"),
        figures=root / out.get("figures_dir", "reports/figures"),
        reports=root / "reports",
    )
