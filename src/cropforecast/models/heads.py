"""Block 5 - Prediction and forecasting heads.

Two heads sit on the graph encoder output, matching the diagram:

``DiseaseClassificationHead``
    Multi-class disease identification at the current time step (38 classes).

``MultiHorizonForecaster``
    Disease risk 1, 3, 5 and 7 days ahead. Each horizon gets both a continuous
    risk value and a low/medium/high band.

The forecaster is deliberately *autoregressive across horizons*: horizon h is
predicted from a shared trunk plus the hidden state carried forward from horizon
h-1. Independent per-horizon heads have no way to know that day 7 follows day 5,
and in practice produce forecasts that jump around non-physically. Carrying state
forward is also what makes the monotonic-uncertainty physics constraint
meaningful.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiseaseClassificationHead(nn.Module):
    """Current-time-step disease classifier."""

    def __init__(self, in_dim: int, num_classes: int = 38, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHorizonForecaster(nn.Module):
    """Predicts risk at each forecast horizon, carrying state between horizons."""

    def __init__(
        self,
        in_dim: int,
        horizons: tuple[int, ...] = (1, 3, 5, 7),
        hidden: int = 128,
        risk_levels: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.horizons = tuple(horizons)
        self.hidden = hidden

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # One GRU cell steps the state from one horizon to the next.
        self.step = nn.GRUCell(hidden, hidden)
        # A learned per-horizon query, so the cell knows how far ahead it is.
        self.horizon_embed = nn.Embedding(len(self.horizons), hidden)

        self.risk_head = nn.Linear(hidden, 1)
        self.level_head = nn.Linear(hidden, risk_levels)
        # Predicted aleatoric spread, used by the monotonic-uncertainty penalty.
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        state = self.trunk(x)

        risks, levels, logvars = [], [], []
        for i in range(len(self.horizons)):
            query = self.horizon_embed(
                torch.full((b,), i, dtype=torch.long, device=x.device)
            )
            state = self.step(query, state)
            # Risk is a probability-like quantity, so squash it to [0, 1].
            risks.append(torch.sigmoid(self.risk_head(state)).squeeze(-1))
            levels.append(self.level_head(state))
            logvars.append(self.logvar_head(state).squeeze(-1))

        return {
            "risk": torch.stack(risks, dim=1),          # (B, H)
            "level_logits": torch.stack(levels, dim=1),  # (B, H, L)
            "logvar": torch.stack(logvars, dim=1),       # (B, H)
        }
