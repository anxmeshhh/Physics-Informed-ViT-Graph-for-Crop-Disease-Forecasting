"""Block 6 - Visual explanations on leaf images.

Two complementary views of what the vision transformer looked at:

``attention_rollout``
    Abadi-style attention rollout: multiply the head-averaged attention matrices
    across all blocks, adding the residual connection at each step, then read the
    CLS row. This shows where the network *routed information from*.

``grad_cam``
    Gradient-weighted activation mapping adapted to transformers. Classic CNN
    Grad-CAM assumes a spatial conv feature map; here the patch tokens of the
    final block are reshaped back onto the patch grid and weighted by the
    gradient of the target logit. This shows what the network *used* for a
    specific prediction, which rollout cannot tell you.

Both return a heatmap normalised to [0, 1] at the original image resolution.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _grid_shape(n_patches: int) -> tuple[int, int]:
    """Infer the square patch grid, e.g. 256 tokens -> 16x16."""
    side = int(round(np.sqrt(n_patches)))
    if side * side != n_patches:
        raise ValueError(f"{n_patches} patch tokens is not a square grid")
    return side, side


def _to_heatmap(grid: torch.Tensor, out_hw: tuple[int, int],
                lo_q: float = 0.02, hi_q: float = 0.98) -> np.ndarray:
    """Upsample a patch-grid map to image size and normalise to [0, 1].

    Normalisation is done between robust quantiles rather than min and max.
    DINOv2 (and ViTs generally) emit a handful of very high-norm "artifact"
    patch tokens that carry global information rather than local content. Under
    plain min-max scaling one such token saturates the map and every genuinely
    informative region collapses to zero - which renders as a flat, empty
    heatmap. Clipping the tails first keeps those outliers from dominating.
    """
    m = grid[None, None].float()
    m = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)[0, 0]

    flat = m.flatten()
    lo = torch.quantile(flat, lo_q)
    hi = torch.quantile(flat, hi_q)
    if float(hi - lo) < 1e-8:            # degenerate map: nothing to show
        lo, hi = flat.min(), flat.max()
    m = ((m - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    return m.detach().cpu().numpy()


@torch.no_grad()
def attention_rollout(
    backbone,
    pixel_values: torch.Tensor,
    discard_ratio: float = 0.55,
    head_fusion: str = "mean",
) -> np.ndarray:
    """CLS attention rollout for one image, as an HxW heatmap in [0, 1]."""
    was_on = backbone.output_attentions
    backbone.output_attentions = True
    backbone.model.config.output_attentions = True
    out = backbone.model(pixel_values=pixel_values, output_attentions=True)
    backbone.output_attentions = was_on

    attentions = out.attentions
    if not attentions:
        raise RuntimeError("Backbone returned no attention maps")

    n_tokens = attentions[0].shape[-1]
    result = torch.eye(n_tokens, device=pixel_values.device)

    for attn in attentions:
        a = attn[0]                                   # (heads, N, N)
        a = a.mean(0) if head_fusion == "mean" else a.max(0).values

        # Drop the weakest links so the rollout is not swamped by noise.
        if discard_ratio > 0:
            flat = a.flatten()
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k).values
                a = torch.where(a < threshold, torch.zeros_like(a), a)

        # Residual connection, then renormalise rows.
        a = a + torch.eye(n_tokens, device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    cls_attention = result[0, 1:]                     # CLS row, patches only
    h, w = _grid_shape(cls_attention.numel())
    return _to_heatmap(cls_attention.reshape(h, w),
                       pixel_values.shape[-2:])


def grad_cam(
    backbone,
    head: torch.nn.Module,
    pixel_values: torch.Tensor,
    target_class: int | None = None,
) -> tuple[np.ndarray, int]:
    """Transformer Grad-CAM for one image. Returns ``(heatmap, class_used)``.

    ``head`` maps a pooled embedding to class logits, so the explanation is of
    the *deployed* classifier rather than of the backbone in isolation.
    """
    backbone.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)

    # Every backbone parameter is frozen, so if the *input* does not require
    # gradient either, autograd builds no graph at all and `retain_grad` raises.
    # Making the pixels require grad is enough to get gradients flowing back to
    # the patch tokens, which is all Grad-CAM needs - and it leaves the frozen
    # weights untouched.
    pixel_values = pixel_values.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        out = backbone.model(pixel_values=pixel_values, output_hidden_states=True)

        # Hook the PENULTIMATE block, not the final hidden state. A CLS-pooled
        # model reads only token 0 downstream, so the gradient of the logit with
        # respect to the *final* layer's patch tokens is identically zero and the
        # CAM comes out perfectly flat. One block earlier, the patch tokens still
        # reach the CLS token through the last attention layer, so they carry
        # real gradient.
        hidden = out.hidden_states
        target_layer = hidden[-2] if len(hidden) >= 2 else out.last_hidden_state
        target_layer.retain_grad()

        final = out.last_hidden_state
        pooled = final[:, 0] if backbone.spec.pool == "cls" else final.mean(1)
        logits = head(pooled)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        logits[0, target_class].backward()

    grads = target_layer.grad                # (1, N, D)
    if grads is None:
        raise RuntimeError("No gradient reached the patch tokens")

    tokens = target_layer
    patch_tokens = tokens[0, 1:] if backbone.spec.pool == "cls" else tokens[0]
    patch_grads = grads[0, 1:] if backbone.spec.pool == "cls" else grads[0]

    # Channel weights are the mean gradient over patches; the CAM is the
    # weighted activation sum.
    weights = patch_grads.mean(dim=0, keepdim=True)          # (1, D)
    signed = (patch_tokens * weights).sum(dim=-1)            # (P,)

    # Transformer token activations are not rectified the way CNN feature maps
    # are, so the weighted sum is frequently negative everywhere. Applying a
    # plain ReLU then yields an all-zero map that normalises to a flat image -
    # which is exactly what a blank Grad-CAM panel means. Keep the positive part
    # when there is one, and otherwise fall back to magnitude, which still shows
    # where the prediction was sensitive.
    cam = F.relu(signed)
    if float(cam.max()) <= 1e-12:
        cam = signed.abs()

    h, w = _grid_shape(cam.numel())
    return _to_heatmap(cam.reshape(h, w), pixel_values.shape[-2:]), target_class


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
            ) -> np.ndarray:
    """Blend a heatmap over an RGB uint8 image using a red-yellow ramp."""
    import matplotlib.cm as cm

    colours = cm.get_cmap("jet")(heatmap)[..., :3]           # (H, W, 3) in [0,1]
    base = image_rgb.astype(np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * colours
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
