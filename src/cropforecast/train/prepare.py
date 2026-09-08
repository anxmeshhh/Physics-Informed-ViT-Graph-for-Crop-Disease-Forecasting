"""Bridge from the observations table to model-ready split tensors.

Builds, for each of train/val/test:
  * the cached vision embeddings for the chosen backbone,
  * scaled climate and metadata matrices (scalers fitted on train only),
  * the raw physical drivers the physics loss needs,
  * an inductive spatio-temporal graph over that split's own observations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.dataset import load_embeddings
from ..data.farms import to_frame
from ..data.splits import leaf_disjoint_split, random_split
from ..features.tabular import (
    attach_site_metadata, fit_feature_space, transform,
)
from ..graph.build import build_graph
from .trainer import SplitTensors


def prepare_splits(
    cfg,
    backbone: str,
    split_mode: str = "leaf_disjoint",
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[dict[str, SplitTensors], dict]:
    """Return ``{"train"/"val"/"test": SplitTensors}`` plus a metadata dict."""
    processed = Path(cfg.paths.processed)
    obs = pd.read_parquet(processed / "observations.parquet")
    obs = attach_site_metadata(obs, to_frame())

    embeddings = load_embeddings(processed, backbone)

    # Embeddings are addressed by image path, not by position. Stage 1 sorts its
    # output by (date, site), so any change to the grounding reshuffles the
    # observations table - and a positional join would then silently pair every
    # leaf with some other leaf's embedding. Extraction is expensive, so the
    # mapping is stored alongside the cache and used here.
    order_file = processed / "embedding_row_order.parquet"
    obs = obs.reset_index(drop=True)
    if order_file.exists():
        order = pd.read_parquet(order_file, columns=["image_path"])
        if len(order) != len(embeddings):
            raise RuntimeError(
                f"embedding_row_order has {len(order)} rows but the cache has "
                f"{len(embeddings)}. Re-run scripts/02_extract_features.py."
            )
        lookup = pd.Series(np.arange(len(order)), index=order.image_path.to_numpy())
        rows = obs.image_path.map(lookup)
        if rows.isna().any():
            missing = int(rows.isna().sum())
            raise RuntimeError(
                f"{missing} observations have no cached embedding. "
                "Re-run scripts/02_extract_features.py."
            )
        obs["_row"] = rows.astype(int).to_numpy()
    else:
        if len(embeddings) != len(obs):
            raise RuntimeError(
                f"Embedding rows ({len(embeddings)}) do not match observations "
                f"({len(obs)}). Re-run scripts/02_extract_features.py."
            )
        obs["_row"] = np.arange(len(obs))

    if split_mode == "leaf_disjoint":
        obs = leaf_disjoint_split(obs, seed=cfg.project.seed)
    elif split_mode == "random":
        obs = random_split(obs, seed=cfg.project.seed)
    else:
        raise ValueError(f"Unknown split_mode {split_mode!r}")

    train_df = obs[obs.split == "train"]
    space = fit_feature_space(train_df)
    sites_df = to_frame()
    horizons = list(cfg.model.heads.horizons)

    out: dict[str, SplitTensors] = {}
    for name in ("train", "val", "test"):
        part = obs[obs.split == name].reset_index(drop=True)
        arrays = transform(part, space, horizons)

        graph = build_graph(
            part, sites_df,
            k_neighbours=cfg.graph.k_neighbours,
            radius_km=cfg.graph.radius_km,
            time_window_days=cfg.graph.time_window_days,
            max_edges_per_node=cfg.graph.max_edges_per_node,
            wind_aware=cfg.graph.wind_aware,
            wind_alignment_threshold=cfg.graph.wind_alignment_threshold,
            min_neighbours=cfg.graph.min_neighbours,
            same_crop=cfg.graph.same_crop,
            seed=cfg.project.seed,
        )
        if verbose:
            print(f"    {name:5s} n={len(part):6,}  {graph.summary()}")

        # Closer edges in space-time carry more weight in the diffusion term.
        dist_norm = graph.edge_attr[:, 0]
        dt_norm = graph.edge_attr[:, 1]
        edge_weight = np.exp(-(dist_norm ** 2 + dt_norm ** 2)).astype(np.float32)

        st = SplitTensors(
            vision=torch.from_numpy(embeddings[part._row.to_numpy()]).float(),
            climate=torch.from_numpy(arrays["climate"]),
            meta=torch.from_numpy(arrays["meta"]),
            physics=torch.from_numpy(arrays["physics"]),
            labels=torch.from_numpy(arrays["labels"]),
            risk_reg=torch.from_numpy(arrays["risk_reg"]),
            risk_cls=torch.from_numpy(arrays["risk_cls"]),
            edge_index=torch.from_numpy(graph.edge_index).long(),
            edge_attr=torch.from_numpy(graph.edge_attr).float(),
            edge_weight=torch.from_numpy(edge_weight),
        )
        out[name] = st.to(device) if device is not None else st

    meta = {
        "vision_dim": embeddings.shape[1],
        "climate_dim": space.climate_dim,
        "meta_dim": out["train"].meta.shape[1],
        "num_classes": int(obs.label.nunique()),
        "horizons": horizons,
        "climate_cols": space.climate_cols,
        "meta_categories": space.meta_categories,
        "feature_space": space,
        "obs": obs,
        "backbone": backbone,
        "split_mode": split_mode,
    }
    return out, meta
