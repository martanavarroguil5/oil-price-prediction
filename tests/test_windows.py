"""Window/target alignment, train-only scaling and look-ahead safety."""

import numpy as np
import torch

from oilcast.features.engineering import compute_features
from oilcast.features.windows import Panel, Scaler, gather_windows
from oilcast.training.splits import purged_train
from oilcast.training.trainer import prepare_data


def test_window_and_target_alignment(panel):
    t, h, w = 3000, 63, panel.window
    x = gather_windows(torch.from_numpy(panel.features), torch.tensor([t]), w)[0].numpy()
    np.testing.assert_array_equal(x, panel.features[t - w + 1 : t + 1])
    assert np.isclose(panel.target(h)[t], np.log(panel.price[t + h] / panel.price[t]))
    assert panel.target_dates(h)[t] == panel.dates.values[t + h]
    assert np.isnan(panel.target(h)[-h:]).all()


def test_valid_mask_requires_full_finite_window(panel):
    valid = panel.valid
    finite = np.isfinite(panel.features).all(axis=1)
    first = int(np.argmax(valid))
    assert finite[first - panel.window + 1 : first + 1].all()
    assert not finite[first - panel.window]


def test_scaler_is_fitted_on_training_rows_only(panel, cfg):
    h = 5
    train = purged_train(panel, h, np.datetime64("2010-01-01"))
    data = prepare_data(panel, h, train, cfg, torch.device("cpu"))
    # perturbing every row after the last training origin must not change the scaler
    feats = panel.features.copy()
    feats[train.max() + 1 :] *= 100
    other = Panel(panel.dates, panel.price, feats, panel.feature_names, panel.window)
    data2 = prepare_data(other, h, train, cfg, torch.device("cpu"))
    for k in ("lo", "hi", "mean", "std"):
        np.testing.assert_allclose(getattr(data.scaler, k), getattr(data2.scaler, k))
    # target scale uses training targets only
    assert np.isclose(data.y_scale, np.std(panel.target(h)[train]))


def test_scaler_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    s = Scaler.fit(x, np.arange(400), (0.01, 0.99))
    s2 = Scaler.from_dict(s.to_dict())
    np.testing.assert_allclose(s.transform(x), s2.transform(x))
    assert np.abs(s.transform(x[:400]).mean(axis=0)).max() < 1e-6


def test_features_do_not_look_ahead(df, cfg):
    t = 4000
    base = compute_features(df, cfg)
    shocked = df.copy()
    shocked.iloc[t + 1 :] *= 3.0
    after = compute_features(shocked, cfg)
    np.testing.assert_allclose(base.iloc[: t + 1].to_numpy(), after.iloc[: t + 1].to_numpy())
    assert not np.allclose(
        base.iloc[t + 1 :].to_numpy(), after.iloc[t + 1 :].to_numpy(), equal_nan=True
    )
