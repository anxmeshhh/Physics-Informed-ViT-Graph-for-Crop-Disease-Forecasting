"""Torch datasets for the image pipeline.

All four backbones consume 224x224 inputs, so images are decoded and resized
exactly **once** and each backbone's own normalisation is applied later on the
GPU. Decoding 54k JPEGs is the real bottleneck; doing it once instead of four
times is what makes the benchmark cheap enough to rerun.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_SIZE = 224


class RawImageDataset(Dataset):
    """Yields uint8 CHW tensors plus the row index, ready for GPU normalisation."""

    def __init__(self, paths: list[str], image_size: int = IMAGENET_SIZE):
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.paths[i]).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        # np.array (not asarray) so the buffer is writable and torch is happy.
        arr = np.array(img, dtype=np.uint8)                # HWC
        return torch.from_numpy(arr).permute(2, 0, 1), i    # CHW


def normalise(batch_u8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
              ) -> torch.Tensor:
    """uint8 CHW batch -> normalised float batch, on whatever device it is on."""
    x = batch_u8.float().div_(255.0)
    return (x - mean) / std


class FusedTabularDataset(Dataset):
    """Cached vision embeddings + climate + metadata, with all supervision targets.

    This is the dataset the fusion / GNN / forecasting stack actually trains on.
    The vision backbone is frozen, so its embeddings are computed once and reused
    - which is what lets the full model train in minutes on a laptop GPU.
    """

    def __init__(
        self,
        embeddings: np.ndarray,        # (N, dim) float32
        climate: np.ndarray,           # (N, n_climate) float32, already scaled
        meta: np.ndarray,              # (N, n_meta) float32, already scaled
        labels: np.ndarray,            # (N,) int64  - disease class
        risk_reg: np.ndarray,          # (N, H) float32 - continuous risk per horizon
        risk_cls: np.ndarray,          # (N, H) int64   - risk band per horizon
        physics: np.ndarray,           # (N, n_phys) float32 - raw physical drivers
    ):
        assert len({len(embeddings), len(climate), len(meta), len(labels)}) == 1
        self.embeddings = torch.from_numpy(embeddings).float()
        self.climate = torch.from_numpy(climate).float()
        self.meta = torch.from_numpy(meta).float()
        self.labels = torch.from_numpy(labels).long()
        self.risk_reg = torch.from_numpy(risk_reg).float()
        self.risk_cls = torch.from_numpy(risk_cls).long()
        self.physics = torch.from_numpy(physics).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "vision": self.embeddings[i],
            "climate": self.climate[i],
            "meta": self.meta[i],
            "label": self.labels[i],
            "risk_reg": self.risk_reg[i],
            "risk_cls": self.risk_cls[i],
            "physics": self.physics[i],
            "index": torch.tensor(i, dtype=torch.long),
        }


def load_embeddings(cache_dir: str | Path, backbone: str) -> np.ndarray:
    """Load a cached embedding matrix written by ``scripts/02_extract_features.py``."""
    path = Path(cache_dir) / f"emb_{backbone}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached embeddings for {backbone!r} at {path}. "
            "Run scripts/02_extract_features.py first."
        )
    return np.load(path).astype(np.float32)
