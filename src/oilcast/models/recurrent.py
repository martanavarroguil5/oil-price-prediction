"""One recurrent network class for the three architectures (simple RNN, LSTM, GRU).

The three variants share everything except the recurrent cell: same input window, same
quantile head and same loss, so the comparison isolates the effect of the cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
ARCH_LABELS = {"rnn": "RNN simple", "lstm": "LSTM", "gru": "GRU"}


@dataclass
class HParams:
    hidden_size: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int

    def to_dict(self) -> dict:
        return asdict(self)


class RecurrentQuantileNet(nn.Module):
    """Recurrent encoder over the input window + linear head emitting one value per quantile."""

    def __init__(
        self,
        cell: str,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_quantiles: int,
        highway: bool = False,
    ) -> None:
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell!r}; expected one of {list(CELLS)}")
        self.rnn = CELLS[cell](
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_quantiles)
        # Ronda 4: atajo lineal desde las variables del último día (componente lineal de
        # LSTNet). Empieza en cero, así que al inicio la red es la misma que sin atajo.
        self.highway = nn.Linear(n_features, n_quantiles) if highway else None
        if self.highway is not None:
            nn.init.zeros_(self.highway.weight)
            nn.init.zeros_(self.highway.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        y = self.head(self.dropout(out[:, -1, :]))
        if self.highway is not None:
            y = y + self.highway(x[:, -1, :])
        return y


def build_model(
    arch: str, n_features: int, hp: HParams, n_quantiles: int, highway: bool = False
) -> RecurrentQuantileNet:
    return RecurrentQuantileNet(
        arch, n_features, hp.hidden_size, hp.num_layers, hp.dropout, n_quantiles, highway
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Mean quantile (pinball) loss. pred: (B, Q), target: (B,), quantiles: (Q,)."""
    diff = target[:, None] - pred
    return torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()
