"""Sliding windows, horizon targets and the train-only scaler.

A sample is identified by its *origin* t (row position on the Brent calendar). Its input is
the feature block ``[t - window + 1, t]`` and its target for horizon h is
``log(P[t + h] / P[t])``, realised on date ``dates[t + h]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from oilcast.config import Config
from oilcast.data.dataset import load_dataset
from oilcast.features.engineering import OPTIONAL_EXTRA, compute_features
from oilcast.paths import Paths


@dataclass
class Panel:
    """Feature matrix plus prices on the Brent calendar."""

    dates: pd.DatetimeIndex
    price: np.ndarray
    features: np.ndarray
    feature_names: list[str]
    window: int
    optional: frozenset[str] = frozenset()  # may be NaN (late-starting extras)

    @classmethod
    def from_dataset(cls, df: pd.DataFrame, cfg: Config) -> Panel:
        feats = compute_features(df, cfg)
        return cls(
            dates=pd.DatetimeIndex(df.index),
            price=df["brent"].to_numpy(dtype=np.float64),
            features=feats.to_numpy(dtype=np.float32),
            feature_names=list(feats.columns),
            window=cfg["features"]["window"],
            optional=frozenset(
                (OPTIONAL_EXTRA | set(cfg["features"].get("optional", []))) & set(feats.columns)
            ),
        )

    @classmethod
    def load(cls, paths: Paths, cfg: Config) -> Panel:
        """Panel from the processed dataset, plus the extra series if the config asks."""
        df = load_dataset(paths)
        if cfg["features"].get("extra"):
            from oilcast.data.extra import load_extra

            df = df.join(load_extra(paths).reindex(df.index))
        return cls.from_dataset(df, cfg)

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def valid(self) -> np.ndarray:
        """True at origins whose whole input window is finite (optional features excluded)."""
        cols = [i for i, n in enumerate(self.feature_names) if n not in self.optional]
        finite = np.isfinite(self.features[:, cols]).all(axis=1).astype(np.int64)
        csum = np.concatenate([[0], np.cumsum(finite)])
        w = self.window
        out = np.zeros(len(self), dtype=bool)
        out[w - 1 :] = (csum[w:] - csum[:-w]) == w
        return out

    def target(self, h: int) -> np.ndarray:
        """log(P[t+h]/P[t]); NaN when t+h is beyond the data."""
        y = np.full(len(self), np.nan)
        lp = np.log(self.price)
        y[:-h] = lp[h:] - lp[:-h]
        return y

    def target_dates(self, h: int) -> np.ndarray:
        td = np.full(len(self), np.datetime64("NaT"), dtype="datetime64[ns]")
        td[:-h] = self.dates.values[h:]
        return td

    def usable(self, h: int) -> np.ndarray:
        """Origins with a complete input window and a realised target."""
        return self.valid & np.isfinite(self.target(h))


@dataclass
class Scaler:
    """Per-feature clipping (train quantiles) followed by standardisation (train mean/std)."""

    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, rows: np.ndarray, quantiles: tuple[float, float]) -> Scaler:
        """Fit on ``features[rows]`` only (rows = boolean mask or indices of training rows)."""
        x = features[rows]
        lo = np.nanquantile(x, quantiles[0], axis=0)
        hi = np.nanquantile(x, quantiles[1], axis=0)
        xc = np.clip(x, lo, hi)
        mean = np.nanmean(xc, axis=0)
        std = np.nanstd(xc, axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return cls(lo=lo, hi=hi, mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        z = (np.clip(features, self.lo, self.hi) - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: getattr(self, k).astype(float).tolist() for k in ("lo", "hi", "mean", "std")}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Scaler:
        return cls(**{k: np.asarray(v, dtype=np.float64) for k, v in d.items()})


def scaler_rows(train_origins: np.ndarray) -> np.ndarray:
    """Rows the scaler may see: everything up to the last training origin (inclusive)."""
    return np.arange(int(train_origins.max()) + 1)


def gather_windows(features: torch.Tensor, origins: torch.Tensor, window: int) -> torch.Tensor:
    """Index (B,) origins into a (T, C) tensor -> (B, window, C) input windows."""
    offsets = torch.arange(-window + 1, 1, device=features.device)
    return features[origins[:, None] + offsets]
