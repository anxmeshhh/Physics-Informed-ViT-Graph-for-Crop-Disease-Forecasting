"""Block 5 - Physics-informed loss.

Implements the diagram's

    Loss = CE Loss + lambda * Physics Loss   (enforces agricultural constraints)

Design principle: **the physics loss never sees the forward epidemiological
model.** If it did, we would be handing the network the very function used to
build the benchmark and the whole exercise would be circular.

Instead each term states a weak, directional fact that any competent agronomist
would sign off on without knowing a single coefficient:

``infection_pressure``
    Forecast risk must not *decrease* as conditions become more conducive to
    infection (more leaf wetness, humidity closer to saturation). Enforced as a
    one-sided penalty on the gradient of predicted risk with respect to the
    conduciveness driver, so only violations are punished.

``dry_suppression``
    Hot, dry air (high vapour pressure deficit) suppresses fungal infection.
    Predicted risk above a ceiling under strongly drying conditions is penalised.

``diffusion``
    Disease risk is spatially smooth: an epidemic does not jump discontinuously
    between two farms 30 km apart observed on the same day. This is a graph
    Dirichlet-energy term, weighted by edge proximity, and it is the term that
    couples the physics loss to the graph structure.

``monotonic_horizon``
    Forecast uncertainty must not shrink as the horizon lengthens. A model
    claiming more confidence about day 7 than day 1 is miscalibrated by
    construction.

Every term is a *soft* penalty: it shapes the solution without ever overriding
the data when the two genuinely disagree.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PhysicsWeights:
    """Relative weight of each constraint inside the physics loss."""

    infection_pressure: float = 1.0
    diffusion: float = 0.5
    monotonic_horizon: float = 0.3
    dry_suppression: float = 0.5


# Column order of the `physics` tensor supplied by the dataset.
PHYSICS_COLUMNS = ("leaf_wetness_hours", "relative_humidity_2m_mean",
                   "vpd_kpa", "temperature_2m_mean")

# Above this VPD the air is strongly drying and fungal infection is unlikely.
VPD_DRY_THRESHOLD = 1.5
# Risk ceiling permitted under those strongly drying conditions.
DRY_RISK_CEILING = 0.55


def infection_pressure_loss(
    risk: torch.Tensor, wetness: torch.Tensor, humidity: torch.Tensor
) -> torch.Tensor:
    """Penalise risk that falls while infection conditions improve.

    Implemented pairwise within the batch: for every pair of observations, if one
    has clearly wetter and more humid conditions than the other, its predicted
    risk should not be lower. Only violations contribute, so the constraint is
    one-sided and never pushes risk upwards on its own.
    """
    mean_risk = risk.mean(dim=1)                       # average over horizons

    # Standardise so the two drivers are comparable.
    w = (wetness - wetness.mean()) / (wetness.std() + 1e-6)
    h = (humidity - humidity.mean()) / (humidity.std() + 1e-6)
    conducive = 0.5 * (w + h)

    # Compare each item against a shuffled partner: O(B) instead of O(B^2).
    perm = torch.randperm(risk.shape[0], device=risk.device)
    d_cond = conducive - conducive[perm]
    d_risk = mean_risk - mean_risk[perm]

    # Only judge pairs whose conditions genuinely differ.
    mask = (d_cond.abs() > 0.5).float()
    # Violation: conditions improved but predicted risk went down.
    violation = F.relu(-d_risk * torch.sign(d_cond)) * mask
    return violation.sum() / (mask.sum() + 1e-6)


def dry_suppression_loss(risk: torch.Tensor, vpd: torch.Tensor) -> torch.Tensor:
    """Penalise high predicted risk when the air is strongly drying."""
    dry = (vpd > VPD_DRY_THRESHOLD).float().unsqueeze(1)   # (B, 1)
    excess = F.relu(risk - DRY_RISK_CEILING) * dry
    return excess.sum() / (dry.sum() * risk.shape[1] + 1e-6)


def diffusion_loss(
    risk: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Graph Dirichlet energy: neighbouring farms should hold similar risk."""
    if edge_index.numel() == 0:
        return risk.new_zeros(())
    src, dst = edge_index[0], edge_index[1]
    diff = (risk[src] - risk[dst]).pow(2).mean(dim=1)     # mean over horizons
    if edge_weight is not None:
        diff = diff * edge_weight
        return diff.sum() / (edge_weight.sum() + 1e-6)
    return diff.mean()


def monotonic_horizon_loss(logvar: torch.Tensor) -> torch.Tensor:
    """Penalise predicted uncertainty that shrinks with a longer horizon."""
    if logvar.shape[1] < 2:
        return logvar.new_zeros(())
    decrease = F.relu(logvar[:, :-1] - logvar[:, 1:])
    return decrease.mean()


def physics_loss(
    risk: torch.Tensor,
    logvar: torch.Tensor,
    physics_feats: torch.Tensor,
    edge_index: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    weights: PhysicsWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total physics penalty plus a per-term breakdown for logging.

    ``physics_feats`` carries the *raw, unscaled* drivers in the order given by
    :data:`PHYSICS_COLUMNS`; the constraints are statements about physical units,
    so they must not be applied to standardised values.
    """
    w = weights or PhysicsWeights()
    wetness = physics_feats[:, 0]
    humidity = physics_feats[:, 1]
    vpd = physics_feats[:, 2]

    terms = {
        "infection_pressure": infection_pressure_loss(risk, wetness, humidity),
        "dry_suppression": dry_suppression_loss(risk, vpd),
        "monotonic_horizon": monotonic_horizon_loss(logvar),
    }
    if edge_index is not None:
        terms["diffusion"] = diffusion_loss(risk, edge_index, edge_weight)
    else:
        terms["diffusion"] = risk.new_zeros(())

    total = sum(getattr(w, name) * value for name, value in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}
