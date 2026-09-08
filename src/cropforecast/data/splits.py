"""Train / validation / test splitting.

PlantVillage photographs each physical leaf several times (2.65 images per leaf
on average, and up to 9 for soybean). A random split therefore puts near-
duplicate views of the *same leaf* on both sides, and the reported accuracy
measures memorisation rather than generalisation.

We provide both split styles so the effect can be measured rather than asserted:

``random``
    The naive baseline everyone reports.
``leaf_disjoint``
    Grouped by ``leaf_group``, so every photograph of a leaf lands in exactly one
    split. This is the honest number.

The official PlantVillage split is *not* leaf-clean either - 134 of its leaf
groups appear on both sides - so ``leaf_disjoint`` is the one to trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def leaf_disjoint_split(
    obs: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    group_col: str = "leaf_group",
    stratify_col: str = "class_name",
) -> pd.DataFrame:
    """Assign each row a ``split`` of train/val/test, never splitting a leaf.

    Groups are stratified by class as far as grouping allows: we stratify the
    *group table* (one row per leaf, carrying that leaf's class) rather than the
    image table, which keeps class proportions close without breaking any group.
    """
    groups = (
        obs.groupby(group_col)[stratify_col].first().reset_index()
        .rename(columns={stratify_col: "cls"})
    )

    # Classes with a single leaf cannot be stratified; pool them.
    counts = groups.cls.value_counts()
    rare = set(counts[counts < 3].index)
    strat = groups.cls.where(~groups.cls.isin(rare), other="__rare__")

    train_groups, holdout_groups = train_test_split(
        groups[group_col], test_size=val_size + test_size,
        random_state=seed, stratify=strat,
    )
    holdout = groups[groups[group_col].isin(holdout_groups)]
    hcounts = holdout.cls.value_counts()
    hrare = set(hcounts[hcounts < 2].index)
    hstrat = holdout.cls.where(~holdout.cls.isin(hrare), other="__rare__")

    val_groups, test_groups = train_test_split(
        holdout[group_col], test_size=test_size / (val_size + test_size),
        random_state=seed, stratify=hstrat,
    )

    mapping = {}
    mapping.update({g: "train" for g in train_groups})
    mapping.update({g: "val" for g in val_groups})
    mapping.update({g: "test" for g in test_groups})

    out = obs.copy()
    out["split"] = out[group_col].map(mapping)
    return out


def random_split(
    obs: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """The leaky baseline: split images without regard to leaf identity."""
    rng = np.random.default_rng(seed)
    r = rng.random(len(obs))
    out = obs.copy()
    out["split"] = np.where(r < 1 - val_size - test_size, "train",
                            np.where(r < 1 - test_size, "val", "test"))
    return out


def leakage_report(obs: pd.DataFrame, group_col: str = "leaf_group") -> dict:
    """How many leaf groups are shared between splits (0 is the goal)."""
    per_group = obs.groupby(group_col).split.nunique()
    shared = per_group[per_group > 1]
    return {
        "n_groups": int(per_group.size),
        "groups_spanning_splits": int(shared.size),
        "images_affected": int(obs[obs[group_col].isin(shared.index)].shape[0]),
    }


def split_summary(obs: pd.DataFrame) -> pd.DataFrame:
    """Row and leaf counts per split."""
    return (
        obs.groupby("split")
        .agg(images=("image_path", "size"),
             leaves=("leaf_group", "nunique"),
             classes=("class_name", "nunique"),
             sites=("site_id", "nunique"))
        .reindex(["train", "val", "test"])
    )
