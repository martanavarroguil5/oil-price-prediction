"""Training loop shared by tuning, walk-forward evaluation and the production fit.

The whole (scaled) feature matrix lives on the device as a (T, C) tensor and mini-batches
are built by indexing origins into it, which is much faster than a DataLoader for
models this small.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from oilcast.config import Config
from oilcast.features.windows import Panel, Scaler, gather_windows, scaler_rows
from oilcast.models.recurrent import HParams, build_model, pinball_loss
from oilcast.utils import set_seed


@dataclass
class PreparedData:
    """Device tensors for one training set (scaler and target scale fitted on it)."""

    features: torch.Tensor
    y: torch.Tensor
    scaler: Scaler
    y_scale: float
    window: int

    @property
    def device(self) -> torch.device:
        return self.features.device


def prepare_data(
    panel: Panel, h: int, train_idx: np.ndarray, cfg: Config, device: torch.device
) -> PreparedData:
    """Fit the input scaler and the target scale on the training origins only."""
    scaler = Scaler.fit(
        panel.features, scaler_rows(train_idx), tuple(cfg["features"]["clip_quantiles"])
    )
    y = panel.target(h)
    y_scale = float(np.std(y[train_idx]))
    # remove_drift: the network learns deviations from the training mean, and predictions are
    # NOT shifted back, so it cannot carry a trend of the training period into the future.
    y_center = float(np.mean(y[train_idx])) if cfg["targets"].get("remove_drift") else 0.0
    y_scaled = np.nan_to_num((y - y_center) / y_scale, nan=0.0).astype(np.float32)
    return PreparedData(
        features=torch.from_numpy(scaler.transform(panel.features)).to(device),
        y=torch.from_numpy(y_scaled).to(device),
        scaler=scaler,
        y_scale=y_scale,
        window=panel.window,
    )


@dataclass
class FitResult:
    model: nn.Module
    epochs_run: int
    best_epoch: int
    best_val: float | None
    seconds: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict_scaled(
    model: nn.Module, data: PreparedData, idx: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    """Raw network output (scaled log-return quantiles, unsorted) for the given origins."""
    model.eval()
    origins = torch.as_tensor(idx, device=data.device)
    out = [
        model(gather_windows(data.features, origins[i : i + batch_size], data.window))
        for i in range(0, len(origins), batch_size)
    ]
    return torch.cat(out).float().cpu().numpy()


def quantile_objective(model, x, target, q):
    return pinball_loss(model(x), target, q)


@torch.no_grad()
def _loss_on(
    model: nn.Module,
    data: PreparedData,
    idx: torch.Tensor,
    q: torch.Tensor,
    batch_size: int,
    objective,
) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        loss = objective(model, gather_windows(data.features, b, data.window), data.y[b], q)
        total += loss.item() * len(b)
    return total / len(idx)


def fit(
    arch: str,
    hp: HParams,
    data: PreparedData,
    train_idx: np.ndarray,
    cfg: Config,
    *,
    epochs: int,
    seed: int,
    val_idx: np.ndarray | None = None,
    on_epoch: Callable[[int, int], None] | None = None,
    model_factory: Callable[[str, int, HParams, int], nn.Module] = build_model,
    objective: Callable = quantile_objective,
) -> FitResult:
    """Train one network. With ``val_idx`` it early-stops on validation pinball loss and
    restores the best epoch; without it, it trains exactly ``epochs`` epochs."""
    tcfg = cfg["training"]
    device = data.device
    quantiles = cfg["targets"]["quantiles"]
    set_seed(seed)
    model = model_factory(arch, data.features.shape[1], hp, len(quantiles)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    q = torch.tensor(quantiles, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    tr = torch.as_tensor(train_idx, device=device)
    va = torch.as_tensor(val_idx, device=device) if val_idx is not None else None

    best_val, best_epoch, best_state = None, epochs, None
    _sync(device)
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen).to(device)]
        for i in range(0, len(perm), hp.batch_size):
            b = perm[i : i + hp.batch_size]
            loss = objective(model, gather_windows(data.features, b, data.window), data.y[b], q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
        if on_epoch:
            on_epoch(epoch, epochs)
        if va is None:
            continue
        v = _loss_on(model, data, va, q, tcfg["eval_batch_size"], objective)
        if best_val is None or v < best_val - 1e-6:
            best_val, best_epoch = v, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= tcfg["patience"] and epoch >= tcfg["min_epochs"]:
            break
    _sync(device)
    seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model, epoch, best_epoch, best_val, seconds)
