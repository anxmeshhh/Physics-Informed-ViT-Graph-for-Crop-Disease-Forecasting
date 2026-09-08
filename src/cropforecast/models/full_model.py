"""The complete architecture, Blocks 3-5 assembled.

    fused = Fusion(vision, climate, metadata)          # Block 3
    z     = GraphEncoder(fused, edge_index, edge_attr) # Block 4
    y     = ClassificationHead(z), Forecaster(z)       # Block 5

The frozen vision backbone (Block 2a) sits outside this module because its
embeddings are cached; :class:`CropDiseaseForecastNet` consumes them directly.
That separation is what lets a full training run finish in minutes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .fusion import FusionModule
from .gnn import build_encoder
from .heads import DiseaseClassificationHead, MultiHorizonForecaster


class CropDiseaseForecastNet(nn.Module):
    """Vision + climate + graph + physics-informed multi-horizon forecaster."""

    def __init__(
        self,
        vision_dim: int,
        climate_dim: int,
        meta_dim: int,
        num_classes: int = 38,
        horizons: tuple[int, ...] = (1, 3, 5, 7),
        fusion_dim: int = 256,
        fusion_mode: str = "gated",
        gnn_conv: str = "sage",
        gnn_hidden: int = 256,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        dropout: float = 0.2,
        gnn_dropout: float = 0.3,
        risk_levels: int = 3,
        edge_dim: int = 4,
    ):
        super().__init__()
        self.horizons = tuple(horizons)

        self.fusion = FusionModule(
            vision_dim, climate_dim, meta_dim,
            hidden_dim=fusion_dim, dropout=dropout, mode=fusion_mode,
        )
        self.encoder = build_encoder(
            gnn_conv, fusion_dim,
            hidden_dim=gnn_hidden, num_layers=gnn_layers,
            heads=gnn_heads, dropout=gnn_dropout, edge_dim=edge_dim,
        )
        self.classifier = DiseaseClassificationHead(
            self.encoder.out_dim, num_classes=num_classes, dropout=dropout,
        )
        self.forecaster = MultiHorizonForecaster(
            self.encoder.out_dim, horizons=self.horizons,
            risk_levels=risk_levels, dropout=dropout,
        )

    def forward(
        self,
        vision: torch.Tensor,
        climate: torch.Tensor,
        meta: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        return_embeddings: bool = False,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        fused = self.fusion(vision, climate, meta)
        z = self.encoder(fused, edge_index, edge_attr, return_attention=return_attention)

        out: dict[str, torch.Tensor] = {"class_logits": self.classifier(z)}
        out.update(self.forecaster(z))
        if return_embeddings:
            out["fused"] = fused
            out["node_embedding"] = z
        return out

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
