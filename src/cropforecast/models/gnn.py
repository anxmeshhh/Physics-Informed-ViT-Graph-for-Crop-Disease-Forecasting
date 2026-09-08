"""Block 4 - Graph neural network module.

Multi-layer message passing over the spatio-temporal farm graph, producing the
"spatially-aware contextual embeddings" of the architecture diagram.

All three encoders named in the diagram are implemented behind one interface:

``sage``
    GraphSAGE. Samples and aggregates neighbours; robust when node degree varies,
    which ours does because production belts are unevenly spaced.
``gcn``
    Symmetric-normalised spectral convolution. The simplest baseline.
``gat``
    Graph attention, and the only one that can *weight* neighbours - so it is the
    one that can learn that a downwind neighbour matters more than an upwind one.
    It is therefore also the encoder whose attention we read back in the
    explainability stage.

Residual connections and layer norm are used throughout: repeated rounds of message
passing over a graph this dense over-smooths badly without them, collapsing every
farm to the same embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GraphEncoder(nn.Module):
    """Stack of message-passing layers with residual connections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        conv: str = "sage",
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int = 4,
        checkpoint: bool | None = None,
    ):
        super().__init__()
        conv = conv.lower()
        if conv not in {"sage", "gcn", "gat"}:
            raise ValueError(f"Unknown conv {conv!r}; use sage | gcn | gat")
        self.conv_type = conv
        self.num_layers = num_layers
        self.dropout = dropout
        # GAT materialises an (E, heads, out_dim) message tensor per layer. On a
        # half-million-edge graph that is ~2 GB a layer, which will not fit in
        # 6 GB alongside the backward pass. Gradient checkpointing recomputes
        # those activations instead of storing them: same maths, same
        # hyperparameters, roughly 30% slower, and it fits.
        self.checkpoint = (conv == "gat") if checkpoint is None else checkpoint

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        dim_in = in_dim
        for _ in range(num_layers):
            if conv == "sage":
                layer = SAGEConv(dim_in, hidden_dim)
            elif conv == "gcn":
                layer = GCNConv(dim_in, hidden_dim)
            else:
                # concat=False averages the heads, keeping the width predictable.
                layer = GATConv(dim_in, hidden_dim, heads=heads, concat=False,
                                dropout=dropout, edge_dim=edge_dim)
            self.convs.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.residual_proj.append(
                nn.Linear(dim_in, hidden_dim) if dim_in != hidden_dim else nn.Identity()
            )
            dim_in = hidden_dim

        self.out_dim = hidden_dim
        self._last_attention: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        attn = None
        for i, (conv, norm, proj) in enumerate(
            zip(self.convs, self.norms, self.residual_proj)
        ):
            identity = proj(x)

            if self.conv_type == "gat":
                if return_attention and i == self.num_layers - 1:
                    h, attn = conv(x, edge_index, edge_attr=edge_attr,
                                   return_attention_weights=True)
                elif self.checkpoint and self.training and x.requires_grad:
                    h = torch_checkpoint(
                        lambda inp, c=conv: c(inp, edge_index, edge_attr=edge_attr),
                        x, use_reentrant=False,
                    )
                else:
                    h = conv(x, edge_index, edge_attr=edge_attr)
            else:
                h = conv(x, edge_index)

            h = norm(h + identity)
            if i < self.num_layers - 1:
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            x = h

        self._last_attention = attn
        return x

    @property
    def last_attention(self):
        """``(edge_index, alpha)`` from the final GAT layer, or None."""
        return self._last_attention


class IdentityEncoder(nn.Module):
    """No-graph ablation: keeps the interface, ignores the edges.

    This is how we measure what the graph is actually worth - without it, any
    gain could just be the extra parameters.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim
        self.conv_type = "none"

    def forward(self, x, edge_index=None, edge_attr=None, return_attention=False):
        return self.net(x)

    @property
    def last_attention(self):
        return None


def build_encoder(conv: str, in_dim: int, **kwargs) -> nn.Module:
    """Factory covering the three GNN variants plus the no-graph ablation."""
    if conv.lower() in {"none", "mlp", "identity"}:
        return IdentityEncoder(in_dim,
                               hidden_dim=kwargs.get("hidden_dim", 256),
                               dropout=kwargs.get("dropout", 0.3))
    return GraphEncoder(in_dim, conv=conv, **kwargs)
