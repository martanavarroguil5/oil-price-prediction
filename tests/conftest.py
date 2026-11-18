"""Shared fixtures: a synthetic dataset with the same schema as the real one (fast, offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oilcast.config import load_config
from oilcast.features.windows import Panel
from oilcast.paths import get_paths


def synthetic_dataset(
    start: str = "2000-01-03", end: str = "2026-06-30", seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)

    def walk(p0: float, vol: float) -> np.ndarray:
        return p0 * np.exp(np.cumsum(rng.normal(0, vol, n)))

    brent = walk(25, 0.02)
    wti = brent * np.exp(rng.normal(-0.03, 0.02, n))
    fut = wti * np.exp(rng.normal(0, 0.005, n))
    df = pd.DataFrame(
        {
            "brent": brent,
            "wti": wti,
            "gas": walk(3, 0.04),
            "ecb_rate": np.round(np.cumsum(rng.normal(0, 0.01, n)) + 2, 2),
            "usd_per_eur": walk(1.1, 0.005),
            "cny_per_usd": walk(7, 0.002),
            "vix": 20 * np.exp(rng.normal(0, 0.2, n)),
            "wti_fut_open": fut,
            "wti_fut_high": fut * 1.01,
            "wti_fut_low": fut * 0.99,
            "wti_fut_close": fut,
            "wti_fut_volume": rng.integers(1e5, 3e5, n).astype(float),
            "copper": walk(3, 0.015),
            "gold": walk(300, 0.01),
            "dxy": walk(100, 0.005),
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )
    return df


@pytest.fixture(scope="session")
def cfg() -> dict:
    c = load_config()
    c["tuning"]["n_trials"] = 1
    c["tuning"]["startup_trials"] = 1
    c["tuning"]["space"] = {
        "hidden_size": [4],
        "num_layers": [1, 1],
        "dropout": [0.0, 0.1],
        "lr": [1e-3, 2e-3],
        "weight_decay": [1e-6, 1e-5],
        "batch_size": [512],
    }
    c["training"].update(max_epochs=1, patience=1, min_epochs=1)
    c["evaluation"]["seeds"] = [0]
    c["evaluation"]["conformal_min_scores"] = 10
    c["final"]["seeds"] = [0]
    return c


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return synthetic_dataset()


@pytest.fixture(scope="session")
def panel(df: pd.DataFrame, cfg: dict) -> Panel:
    return Panel.from_dataset(df, cfg)


@pytest.fixture
def tmp_paths(tmp_path, df):
    paths = get_paths({}, root=tmp_path)
    paths.ensure()
    df.to_csv(paths.dataset_csv)
    return paths
