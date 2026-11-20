"""Load the production artifacts and forecast the three horizons from the latest data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oilcast.config import Config, horizon_config
from oilcast.features.windows import Panel, Scaler, gather_windows
from oilcast.inference.calendar import add_trading_days
from oilcast.models.recurrent import HParams, build_model
from oilcast.utils import read_json


@dataclass
class HorizonForecast:
    horizon: int
    origin_date: pd.Timestamp
    target_date: pd.Timestamp
    last_price: float
    median: float
    quantiles: dict[float, float]  # raw network quantiles, as prices
    intervals: dict[str, tuple[float, float]]  # calibrated intervals, as prices

    @property
    def change(self) -> float:
        return self.median - self.last_price

    @property
    def change_pct(self) -> float:
        return self.median / self.last_price - 1


class Forecaster:
    """Seed-ensemble forecaster for every horizon in ``models/``."""

    def __init__(self, models_dir: Path, device: str = "cpu") -> None:
        self.meta = read_json(models_dir / "metadata.json")
        self.device = torch.device(device)
        self.quantiles = self.meta["quantiles"]
        self.horizons: dict[int, dict] = {}
        for hs in self.meta["spec"]["horizons"]:
            art = torch.load(models_dir / f"h{hs}.pt", map_location=self.device, weights_only=True)
            hp = HParams(**art["hparams"])
            names = art.get("feature_names", self.meta["feature_names"])
            nets = []
            for state in art["states"]:
                net = build_model(
                    art["arch"], len(names), hp, len(self.quantiles), art.get("highway", False)
                )
                net.load_state_dict(state)
                nets.append(net.to(self.device).eval())
            self.horizons[int(hs)] = {
                "nets": nets,
                "feature_names": names,
                "spec": self.meta["spec"]["horizons"][hs],
                "scaler": Scaler.from_dict(art["scaler"]),
                "y_scale": float(art["y_scale"]),
                "offsets": self.meta["spec"]["horizons"][hs]["conformal_offsets"],
                "shrink": float(self.meta["spec"]["horizons"][hs].get("center_shrink", 1.0)),
            }

    @torch.no_grad()
    def predict_returns(self, panel: Panel, h: int, origins: np.ndarray) -> np.ndarray:
        """Sorted ensemble log-return quantiles (n, Q) for arbitrary origins."""
        art = self.horizons[h]
        if panel.feature_names != art["feature_names"]:
            raise ValueError("feature set differs from the one the model was trained with")
        feats = torch.from_numpy(art["scaler"].transform(panel.features)).to(self.device)
        x = gather_windows(feats, torch.as_tensor(origins, device=self.device), panel.window)
        out = torch.stack([net(x) for net in art["nets"]]).cpu().numpy()
        q = np.sort(out, axis=2).mean(axis=0) * art["y_scale"]
        return np.sort(q, axis=1)

    def panel(self, df: pd.DataFrame, cfg: Config, h: int) -> Panel:
        """Input panel of horizon ``h`` (its own variables when the spec sets them)."""
        return Panel.from_dataset(df, horizon_config(cfg, self.horizons[h]["spec"]))

    def forecast(self, df: pd.DataFrame, cfg: Config) -> dict[int, HorizonForecast]:
        """Forecast from the last row of the dataset (the latest real Brent close)."""
        out = {}
        for h, art in self.horizons.items():
            panel = self.panel(df, cfg, h)
            origin = len(panel) - 1
            if not panel.valid[origin]:
                raise ValueError("incomplete input window at the last date")
            last_price = float(panel.price[origin])
            origin_date = panel.dates[origin]
            q = self.predict_returns(panel, h, np.array([origin]))[0]
            # center recalibration: shift the distribution so its median is shrink * median
            q = q + (art["shrink"] - 1.0) * float(np.interp(0.5, self.quantiles, q))
            qmap = dict(zip(self.quantiles, q, strict=True))
            intervals = {}
            for level, (ql, qh) in self.meta["intervals"].items():
                off = art["offsets"].get(level, 0.0)
                intervals[level] = (
                    last_price * float(np.exp(qmap[ql] - off)),
                    last_price * float(np.exp(qmap[qh] + off)),
                )
            out[h] = HorizonForecast(
                horizon=h,
                origin_date=origin_date,
                target_date=add_trading_days(origin_date, h),
                last_price=last_price,
                median=last_price * float(np.exp(qmap[0.5])),
                quantiles={k: last_price * float(np.exp(v)) for k, v in qmap.items()},
                intervals=intervals,
            )
        return out
