"""Ronda 4: variable oro/crudo, subconjunto de variables y atajo lineal."""

from __future__ import annotations

import copy

import numpy as np
import torch

from oilcast.features.windows import Panel
from oilcast.models.recurrent import HParams, build_model


def test_gold_oil_feature_is_deviation_from_its_mean(df, cfg) -> None:
    c = copy.deepcopy(cfg)
    c["features"].update(valuation=True, valuation_window=300, only=["oro_crudo"])
    p = Panel.from_dataset(df, c)
    assert p.feature_names == ["oro_crudo"]
    ratio = np.log(df["gold"]) - np.log(df["brent"])
    expected = ratio - ratio.rolling(300, min_periods=252).mean()
    np.testing.assert_allclose(p.features[:, 0], expected.to_numpy(), rtol=1e-4, equal_nan=True)
    # constant ratio -> no signal
    flat = df.assign(gold=df["brent"] * 20.0)
    q = Panel.from_dataset(flat, c)
    assert np.nanmax(np.abs(q.features[:, 0])) < 1e-5


def test_highway_starts_as_plain_network() -> None:
    hp = HParams(8, 1, 0.0, 1e-3, 0.0, 32)
    torch.manual_seed(0)
    plain = build_model("gru", 3, hp, 7)
    torch.manual_seed(0)
    hw = build_model("gru", 3, hp, 7, highway=True)
    x = torch.randn(5, 60, 3)
    plain.eval()
    hw.eval()
    assert torch.allclose(plain(x), hw(x))
    hw.highway.weight.data.fill_(1.0)
    assert not torch.allclose(plain(x), hw(x))


def test_horizon_config_overrides_only_that_horizon(cfg) -> None:
    from oilcast.config import horizon_config

    assert horizon_config(cfg, {"hparams": {}}) is cfg
    hc = horizon_config(
        cfg, {"features": {"valuation": True, "only": ["oro_crudo"]}, "highway": True}
    )
    assert hc["features"]["only"] == ["oro_crudo"] and hc["model"]["highway"] is True
    assert hc["features"]["window"] == cfg["features"]["window"]
    assert "only" not in cfg["features"]  # the global config is untouched
