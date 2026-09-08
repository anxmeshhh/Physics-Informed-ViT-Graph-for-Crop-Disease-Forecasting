"""Block 6 - Feature attribution and graph explainability.

``integrated_gradients``
    Captum Integrated Gradients over the climate stream, attributing a forecast
    back to individual weather variables. IG is the right tool here because it
    satisfies completeness - the attributions sum to the difference between the
    prediction and a baseline - so "which variable drove this forecast" has an
    exact answer rather than a heuristic one.

``shap_climate``
    Global feature importance over the climate block via SHAP on a surrogate.
    Answers "what does the model rely on in general", where IG answers "why this
    particular field, today".

``explain_graph``
    Edge and neighbour importance. With a GAT encoder the learned attention is
    read directly; otherwise importance is the gradient of the prediction with
    respect to each edge's contribution. This is what surfaces "risk here is
    being driven by that upwind farm 60 km away".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


class _ClimateOnly(torch.nn.Module):
    """Wraps the full model so only the climate stream varies.

    Attribution needs a function of one tensor; everything else is pinned to the
    observation being explained.
    """

    def __init__(self, model, vision, meta, edge_index=None, edge_attr=None,
                 target: str = "risk", horizon: int = 0):
        super().__init__()
        self.model = model
        self.register_buffer("vision", vision.detach())
        self.register_buffer("meta", meta.detach())
        # An empty (2, 0) edge index, not None: the conv layers index into this
        # tensor unconditionally, so None crashes them. With no edges every node
        # simply falls back to its self-transform, which is what we want here.
        self.edge_index = (
            edge_index if edge_index is not None
            else torch.zeros((2, 0), dtype=torch.long, device=vision.device)
        )
        self.edge_attr = edge_attr
        self.target = target
        self.horizon = horizon

    def forward(self, climate: torch.Tensor) -> torch.Tensor:
        # Attribution perturbs a batch of arbitrary size; broadcast the pinned
        # streams to match rather than assuming they already line up.
        n = climate.shape[0]
        vision = self.vision[:n] if self.vision.shape[0] >= n else \
            self.vision[:1].expand(n, -1)
        meta = self.meta[:n] if self.meta.shape[0] >= n else \
            self.meta[:1].expand(n, -1)

        out = self.model(vision, climate, meta, self.edge_index, self.edge_attr)
        if self.target == "risk":
            return out["risk"][:, self.horizon]
        return out["class_logits"]


def integrated_gradients(
    model,
    vision: torch.Tensor,
    climate: torch.Tensor,
    meta: torch.Tensor,
    climate_cols: list[str],
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
    horizon: int = 0,
    n_steps: int = 64,
) -> pd.DataFrame:
    """Attribute a risk forecast to each climate variable.

    The baseline is the all-zeros vector, which in standardised space is the
    training mean - i.e. "average weather". Attributions therefore read as
    "how much did this field's weather differing from normal move the forecast".
    """
    from captum.attr import IntegratedGradients

    # Graph edges are dropped for attribution: IG perturbs one node's features,
    # and message passing would smear that perturbation across neighbours,
    # attributing this field's forecast to its neighbours' weather.
    wrapper = _ClimateOnly(model, vision, meta, None, None,
                           target="risk", horizon=horizon).eval()
    ig = IntegratedGradients(wrapper)

    attributions, delta = ig.attribute(
        climate, baselines=torch.zeros_like(climate),
        n_steps=n_steps, return_convergence_delta=True,
    )
    attr = attributions.detach().cpu().numpy()

    df = pd.DataFrame({
        "feature": climate_cols,
        "attribution": attr.mean(axis=0),
        "abs_attribution": np.abs(attr).mean(axis=0),
        "value_mean": climate.detach().cpu().numpy().mean(axis=0),
    })
    df.attrs["convergence_delta"] = float(delta.abs().mean())
    return df.sort_values("abs_attribution", ascending=False).reset_index(drop=True)


def shap_climate(
    model,
    vision: torch.Tensor,
    climate: torch.Tensor,
    meta: torch.Tensor,
    climate_cols: list[str],
    horizon: int = 0,
    n_background: int = 100,
    n_explain: int = 200,
) -> pd.DataFrame:
    """Global SHAP importance over the climate block (GradientExplainer)."""
    import shap

    wrapper = _ClimateOnly(model, vision[:n_background], meta[:n_background],
                           None, None, target="risk", horizon=horizon).eval()

    background = climate[:n_background]
    explain = climate[:min(n_explain, len(climate))]

    explainer = shap.GradientExplainer(wrapper, background)
    values = explainer.shap_values(explain)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values).reshape(len(explain), -1)

    return (
        pd.DataFrame({
            "feature": climate_cols,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_graph(
    model,
    batch,
    node_id: int,
    obs: pd.DataFrame,
    horizon: int = 0,
    top_k: int = 10,
) -> pd.DataFrame:
    """Rank the neighbours influencing one node's forecast.

    Uses gradient-based edge attribution: the magnitude of the gradient of this
    node's predicted risk with respect to each neighbour's fused representation.
    """
    model.eval()
    edge_index = batch.edge_index

    incoming = (edge_index[1] == node_id).nonzero(as_tuple=True)[0]
    if incoming.numel() == 0:
        return pd.DataFrame(columns=["source_node", "site_id", "date",
                                     "importance", "dist_km_norm", "is_downwind"])

    vision = batch.vision.clone().requires_grad_(True)
    out = model(vision, batch.climate, batch.meta, edge_index, batch.edge_attr)
    out["risk"][node_id, horizon].backward()

    # Sensitivity of this node's forecast to each source node's input.
    node_sensitivity = vision.grad.abs().sum(dim=1)

    sources = edge_index[0, incoming].detach().cpu().numpy()
    importance = node_sensitivity[edge_index[0, incoming]].detach().cpu().numpy()
    attrs = batch.edge_attr[incoming].detach().cpu().numpy()

    rows = pd.DataFrame({
        "source_node": sources,
        "importance": importance,
        "dist_km_norm": attrs[:, 0],
        "dt_days_norm": attrs[:, 1],
        "wind_alignment": attrs[:, 2],
        "is_downwind": attrs[:, 3].astype(bool),
    })
    if {"site_id", "date"}.issubset(obs.columns):
        rows["site_id"] = obs.site_id.to_numpy()[sources]
        rows["date"] = obs.date.to_numpy()[sources]

    total = rows.importance.sum()
    rows["importance_pct"] = 100.0 * rows.importance / max(total, 1e-9)
    return rows.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


def gat_attention(model, batch, obs: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Read learned attention off the final GAT layer, if the encoder is a GAT."""
    model.eval()
    with torch.no_grad():
        model(batch.vision, batch.climate, batch.meta,
              batch.edge_index, batch.edge_attr, return_attention=True)

    attn = model.encoder.last_attention
    if attn is None:
        raise RuntimeError("Encoder is not a GAT, or attention was not returned")

    edge_index, alpha = attn
    alpha = alpha.mean(dim=1).detach().cpu().numpy()   # average the heads
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()

    rows = pd.DataFrame({"source": src, "target": dst, "attention": alpha})
    if "site_id" in obs.columns:
        sites = obs.site_id.to_numpy()
        n = len(sites)
        # GAT adds self-loops, so indices can exceed the observation count.
        valid = (rows.source < n) & (rows.target < n)
        rows = rows[valid]
        rows["source_site"] = sites[rows.source.to_numpy()]
        rows["target_site"] = sites[rows.target.to_numpy()]
    return rows.sort_values("attention", ascending=False).head(top_k).reset_index(drop=True)
