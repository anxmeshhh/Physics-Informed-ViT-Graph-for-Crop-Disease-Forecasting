"""Configuration loading and global paths.

Every stage imports :func:`load_config` so that a single YAML edit propagates
through the whole pipeline. Paths in the YAML are relative to the project root
and are resolved to absolute paths here.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# src/cropforecast/config.py -> project root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

_PATH_KEYS = {"raw_images", "splits", "leaf_map", "interim", "processed",
              "climate", "checkpoints", "figures", "reports"}


class Config(dict):
    """A dict that also supports attribute access, so cfg.graph.k_neighbours works.

    Nested dicts are converted to Config **once, at construction**. An earlier
    version wrapped them lazily on every attribute access, which silently
    returned a fresh copy each time - so ``cfg.graph["radius_km"] = 500`` wrote
    into a temporary and the change never reached the real config. Any sweep
    written against that API quietly evaluated the same setting over and over.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Config):
                super().__setitem__(key, Config(value))

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config(value) if isinstance(value, dict) else value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, Config(value) if isinstance(value, dict) else value)


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config and resolve every declared path to an absolute one."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    for key, value in raw.get("paths", {}).items():
        if key in _PATH_KEYS and value is not None:
            raw["paths"][key] = str((PROJECT_ROOT / value).resolve())

    return Config(raw)


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches, so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(cfg: Config) -> None:
    """Create every output directory the pipeline writes to."""
    for key in ("interim", "processed", "climate", "checkpoints", "figures", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Device:
    """Resolved compute device plus the AMP flag, chosen once and reused."""

    name: str
    amp: bool

    @staticmethod
    def auto(use_amp: bool = True) -> "Device":
        import torch

        if torch.cuda.is_available():
            return Device("cuda", use_amp)
        return Device("cpu", False)   # AMP on CPU is not worth it

    def torch(self):
        import torch

        return torch.device(self.name)
