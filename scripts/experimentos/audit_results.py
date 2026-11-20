"""Audit saved dates, metrics and year sensitivity without fitting or selecting models."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from oilcast.config import load_config
from oilcast.evaluation.metrics import point_metrics
from oilcast.features.windows import Panel
from oilcast.paths import get_paths
from oilcast.training.splits import cv_folds, refit_blocks
from oilcast.utils import write_json


def main() -> None:
    cfg = load_config()
    paths = get_paths(cfg)
    panel = Panel.load(paths, cfg)
    comparison = pd.read_csv(paths.results / "comparison.csv")
    checks, stability = [], []
    inputs = [paths.dataset_csv, paths.results / "comparison.csv"]
    for h in cfg["targets"]["horizons"]:
        folds = cv_folds(panel, h, cfg)
        blocks = refit_blocks(panel, h, cfg)
        for fold in folds:
            assert (
                panel.target_dates(h)[fold.train] < np.datetime64(panel.dates[fold.val].min())
            ).all()
        for block in blocks:
            assert (panel.target_dates(h)[block.train] < np.datetime64(block.refit_date)).all()
        reference = None
        for arch in cfg["architectures"]:
            path = paths.results / "predictions" / f"{arch}_h{h}_ensemble.csv"
            inputs.append(path)
            frame = pd.read_csv(path, parse_dates=["origin", "target_date", "refit_date"])
            assert not frame.origin.duplicated().any()
            assert (frame.refit_date <= frame.origin).all()
            positions = panel.dates.get_indexer(frame.origin)
            assert (positions >= 0).all()
            np.testing.assert_array_equal(
                panel.target_dates(h)[positions], frame.target_date.to_numpy()
            )
            np.testing.assert_allclose(panel.price[positions], frame.price, rtol=3e-5)
            np.testing.assert_allclose(panel.target(h)[positions], frame.y, rtol=3e-5, atol=1e-6)
            assert np.isfinite(frame[["price", "y", "q50", "lo80", "hi80"]]).all().all()
            assert (frame.lo80 <= frame.hi80).all()
            test = frame[frame.phase == "test"].copy()
            if reference is not None:
                np.testing.assert_array_equal(reference, test.origin)
            reference = test.origin
            actual = test.price.to_numpy() * np.exp(test.y.to_numpy())
            predicted = test.price.to_numpy() * np.exp(test.q50.to_numpy())
            calculated = point_metrics(actual, predicted)
            expected = comparison[(comparison.model == arch) & (comparison.horizon == h)].iloc[0]
            keys = ["mae", "rmse", "r2"]
            np.testing.assert_allclose(
                [calculated[k] for k in keys], [expected[k] for k in keys], rtol=1e-5, atol=1e-5
            )
            em, en = np.abs(predicted - actual), np.abs(test.price.to_numpy() - actual)
            years = test.origin.dt.year.to_numpy()
            unique_years = sorted(set(years))
            annual = [1 - em[years == y].mean() / en[years == y].mean() for y in unique_years]
            excluding = [1 - em[years != y].mean() / en[years != y].mean() for y in unique_years]
            stability.append(
                {
                    "model": arch,
                    "horizon": h,
                    "pooled_skill_pct": 100 * (1 - em.mean() / en.mean()),
                    "equal_year_skill_pct": 100 * np.mean(annual),
                    "positive_years": int(sum(s > 0 for s in annual)),
                    "years": len(annual),
                    "exclude_one_year_min_pct": 100 * min(excluding),
                    "exclude_one_year_max_pct": 100 * max(excluding),
                }
            )
            roundtrip = frame.price.to_numpy() * np.exp(frame.y.to_numpy())
            checks.append(
                {
                    "model": arch,
                    "horizon": h,
                    "n_test": len(test),
                    "purged_cv_folds": len(folds),
                    "purged_refit_blocks": len(blocks),
                    "price_target_roundtrip_max_usd": float(
                        np.max(np.abs(panel.price[positions + h] - roundtrip))
                    ),
                }
            )
    pd.DataFrame(stability).to_csv(
        paths.reports / "experiments" / "auditoria" / "estabilidad.csv", index=False
    )
    write_json(
        paths.reports / "experiments" / "auditoria" / "validacion.json",
        {
            "scope": (
                "saved predictions, planned temporal splits, metric reconstruction "
                "and year sensitivity"
            ),
            "limitations": [
                "not a new independent test",
                "no reconstruction of publication timestamps or historical data vintages",
            ],
            "checks": checks,
            "input_sha256": {
                p.relative_to(paths.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in inputs
            },
        },
    )
    print("Audit completed: reports/experiments/auditoria/")


if __name__ == "__main__":
    main()
