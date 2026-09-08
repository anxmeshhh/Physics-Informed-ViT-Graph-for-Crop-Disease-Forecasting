"""Block 3 - Feature fusion.

Combines the three input streams of the architecture diagram

    Vision Embeddings  (+)  Climate Features  (+)  Farm Metadata

into a single node representation for the graph encoder.

Two fusion strategies are provided:

``concat``
    The diagram's own "Concatenation / Projection": each stream is projected to
    a common width, concatenated, and passed through an MLP.

``gated``
    A FiLM-style variant where the climate stream modulates the vision stream
    multiplicatively. This matters agronomically: the *same* visual lesion means
    something different under a fortnight of leaf wetness than it does in a dry
    spell, and a purely additive concat cannot express that interaction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StreamEncoder(nn.Module):
    """Projects one input stream to the shared fusion width."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModule(nn.Module):
    """Vision + climate + metadata -> one fused node feature vector."""

    def __init__(
        self,
        vision_dim: int,
        climate_dim: int,
        meta_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        mode: str = "gated",
    ):
        super().__init__()
        if mode not in {"concat", "gated"}:
            raise ValueError(f"Unknown fusion mode {mode!r}")
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.vision_enc = StreamEncoder(vision_dim, hidden_dim, dropout)
        self.climate_enc = StreamEncoder(climate_dim, hidden_dim, dropout)
        self.meta_enc = StreamEncoder(meta_dim, hidden_dim, dropout)

        if mode == "gated":
            # Climate produces a scale and shift applied to the vision stream.
            self.film = nn.Linear(hidden_dim, 2 * hidden_dim)

        self.mix = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, vision: torch.Tensor, climate: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        v = self.vision_enc(vision)
        c = self.climate_enc(climate)
        m = self.meta_enc(meta)

        if self.mode == "gated":
            gamma, beta = self.film(c).chunk(2, dim=-1)
            # 1 + gamma keeps the transform near identity at initialisation.
            v = v * (1.0 + gamma) + beta

        return self.mix(torch.cat([v, c, m], dim=-1))
