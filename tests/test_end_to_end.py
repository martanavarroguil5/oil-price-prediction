"""Tiny end-to-end run on synthetic data: tune -> evaluate -> summarize -> final -> forecast."""

import json

import numpy as np
import pytest

from oilcast.data.dataset import load_dataset
from oilcast.evaluation.summary import summarize
from oilcast.inference.forecaster import Forecaster
from oilcast.training.benchmark import run_timing
from oilcast.training.evaluate import run_evaluation
from oilcast.training.final import spec_from_metadata, spec_from_results, train_final
from oilcast.training.tune import run_study


@pytest.fixture
def small_cfg(cfg):
    c = {**cfg, "architectures": ["gru"], "targets": {**cfg["targets"], "horizons": [5]}}
    return c


def test_pipeline_end_to_end(small_cfg, tmp_paths, monkeypatch):
    monkeypatch.setenv("OILCAST_DEVICE", "cpu")
    cfg, paths = small_cfg, tmp_paths
    best = run_study(cfg, paths, "gru", 5)
    assert best["n_trials"] == 1 and best["final_epochs"] >= 1
    pred = run_evaluation(cfg, paths, "gru", 5)
    assert {"calibration", "test"} == set(pred["phase"])
    run_timing(cfg, paths)
    sel = summarize(cfg, paths)
    assert sel["chosen"] == "gru"
    meta = train_final(cfg, paths, spec_from_results(cfg, paths))
    assert (paths.models / "h5.pt").exists() and meta["arch"] == "gru"

    df = load_dataset(paths)
    fc = Forecaster(paths.models).forecast(df, cfg)[5]
    lo, hi = fc.intervals["80"]
    assert lo <= fc.median <= hi and fc.last_price == df["brent"].iloc[-1]
    assert fc.target_date > fc.origin_date

    # look-ahead: adding a future row with an extreme price must not change the forecast
    # issued at the previous origin
    fcst = Forecaster(paths.models)
    from oilcast.features.windows import Panel

    p1 = Panel.from_dataset(df, cfg)
    df2 = df.copy()
    df2.iloc[-50:] *= 5
    p2 = Panel.from_dataset(df2, cfg)
    origin = np.array([len(df) - 51])
    np.testing.assert_allclose(
        fcst.predict_returns(p1, 5, origin), fcst.predict_returns(p2, 5, origin), rtol=1e-5
    )

    # retraining from the stored spec (what the app does) works without the results dir
    meta2 = train_final(cfg, paths, spec_from_metadata(paths))
    assert json.dumps(meta2["spec"], sort_keys=True) == json.dumps(meta["spec"], sort_keys=True)
