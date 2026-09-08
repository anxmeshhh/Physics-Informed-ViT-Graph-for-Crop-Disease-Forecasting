"""Block 2a - Vision backbones.

One interface over the four transformer families the related-work table
compares, so the benchmark is genuinely apples-to-apples:

======  ==================================  ====  ==========================
key     checkpoint                          dim   family
======  ==================================  ====  ==========================
dinov2  facebook/dinov2-small                384  self-supervised ViT (primary)
vit     google/vit-base-patch16-224-in21k    768  supervised ViT
swin    microsoft/swin-tiny-patch4-window7   768  hierarchical shifted-window
deit    facebook/deit-small-patch16-224      384  distilled ViT
======  ==================================  ====  ==========================

The architecture diagram specifies a **frozen** backbone, so by default the
weights are frozen and only the fusion/GNN/head stack is trained. That is also
what makes the pipeline trainable on a 6 GB laptop GPU: embeddings are extracted
once, cached to disk, and reused by every downstream experiment.

Pooling differs by family and the difference is honest rather than papered over:
ViT/DeiT/DINOv2 expose a CLS token, Swin does not and is mean-pooled. Each
backbone reports which rule it used.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModel


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    hf_id: str
    dim: int
    pool: str          # "cls" or "mean"
    family: str


SPECS: dict[str, BackboneSpec] = {
    "dinov2": BackboneSpec("dinov2", "facebook/dinov2-small", 384, "cls",
                           "Self-supervised ViT (DINOv2)"),
    "vit":    BackboneSpec("vit", "google/vit-base-patch16-224-in21k", 768, "cls",
                           "Supervised ViT-B/16"),
    "swin":   BackboneSpec("swin", "microsoft/swin-tiny-patch4-window7-224", 768, "mean",
                           "Hierarchical Swin-T"),
    "deit":   BackboneSpec("deit", "facebook/deit-small-patch16-224", 384, "cls",
                           "Distilled DeiT-S"),
}


def build_transform(hf_id: str, image_size: int = 224, train: bool = False) -> T.Compose:
    """Preprocessing matched to the checkpoint's own normalisation statistics.

    Using each checkpoint's real mean/std matters: feeding ImageNet statistics to
    DINOv2 (or vice versa) quietly costs a couple of points of accuracy.
    """
    proc = AutoImageProcessor.from_pretrained(hf_id)
    mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(proc, "image_std", [0.229, 0.224, 0.225])

    if train:
        # Mild, label-preserving augmentation. No vertical flip or heavy colour
        # jitter: lesion colour and leaf orientation are diagnostic signal.
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class VisionBackbone(nn.Module):
    """A transformer image encoder with a uniform output contract."""

    def __init__(self, key: str, frozen: bool = True, output_attentions: bool = False):
        super().__init__()
        if key not in SPECS:
            raise KeyError(f"Unknown backbone {key!r}; choose from {sorted(SPECS)}")
        self.spec = SPECS[key]
        self.model = AutoModel.from_pretrained(
            self.spec.hf_id, attn_implementation="eager" if output_attentions else None
        )
        self.output_attentions = output_attentions
        self.frozen = frozen
        if frozen:
            self.freeze()

    # -- properties ---------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.spec.dim

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):     # keep a frozen backbone in eval mode
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    # -- forward ------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns ``pooled`` (B, dim), ``tokens`` (B, N, dim) and maybe ``attentions``."""
        out = self.model(pixel_values=pixel_values,
                         output_attentions=self.output_attentions)
        tokens = out.last_hidden_state                     # (B, N, dim)

        if self.spec.pool == "cls":
            pooled = tokens[:, 0]                          # CLS token
        else:
            pooled = tokens.mean(dim=1)                    # Swin has no CLS token

        result = {"pooled": pooled, "tokens": tokens}
        if self.output_attentions and getattr(out, "attentions", None) is not None:
            result["attentions"] = out.attentions[-1]      # last block, (B, H, N, N)
        return result

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Pooled embedding only - the path used for cached feature extraction."""
        return self.forward(pixel_values)["pooled"]


def load_backbone(key: str, frozen: bool = True, output_attentions: bool = False
                  ) -> VisionBackbone:
    """Convenience constructor."""
    return VisionBackbone(key, frozen=frozen, output_attentions=output_attentions)


def available() -> list[str]:
    return sorted(SPECS)
