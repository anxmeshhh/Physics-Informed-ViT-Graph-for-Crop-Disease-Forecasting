"""Block 1 - PlantVillage image index.

Builds one master DataFrame describing every leaf image: file path, crop,
disease, class label, official split, and - importantly - the *leaf group*.

The leaf group matters. PlantVillage photographs the same physical leaf from
several angles, so a naive random split leaks near-duplicate images from train
into test and inflates accuracy by a wide margin. The published
``leaf-map.json`` tells us which files belong to the same leaf; we carry that
through as ``leaf_group`` and use it as the grouping key for every split.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# PlantVillage encodes "healthy" with a few different spellings across crops.
HEALTHY_TOKENS = {"healthy"}


def parse_class_name(class_dir: str) -> tuple[str, str]:
    """Split a PlantVillage folder name into ``(crop, disease)``.

    ``Tomato___Late_blight`` -> ``("Tomato", "Late_blight")``
    """
    if "___" in class_dir:
        crop, disease = class_dir.split("___", 1)
    else:  # defensive: folder without the separator
        crop, disease = class_dir, "unknown"
    return crop, disease


def _leaf_key(path: Path) -> str:
    """The ``leaf-map.json`` key for an image path.

    Filenames look like ``<uuid>___<CAPTURE_ID>.JPG``; the capture id, lowercased,
    is the leaf-map key.
    """
    stem = path.stem
    return stem.split("___", 1)[1].lower() if "___" in stem else stem.lower()


def load_leaf_map(leaf_map_path: str | Path) -> dict[str, str]:
    """Map each leaf key to a stable ``"<class>:::<group id>"`` string."""
    with open(leaf_map_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # Values are single-element lists like ["Peach___healthy:::54.0"].
    return {k: (v[0] if isinstance(v, list) and v else str(v)) for k, v in raw.items()}


def _load_split(split_path: Path) -> set[str]:
    """Read a split file into a set of basenames for fast membership tests."""
    if not split_path.exists():
        return set()
    with open(split_path, "r", encoding="utf-8") as fh:
        return {Path(line.strip()).name for line in fh if line.strip()}


def build_index(
    raw_images: str | Path,
    splits_dir: str | Path,
    leaf_map_path: str | Path,
    max_per_class: int | None = None,
) -> pd.DataFrame:
    """Index every PlantVillage colour image into a tidy DataFrame.

    Returns columns:
        image_path, filename, class_name, crop, disease, is_healthy,
        leaf_group, split, label
    """
    raw_images = Path(raw_images)
    splits_dir = Path(splits_dir)
    if not raw_images.exists():
        raise FileNotFoundError(f"PlantVillage images not found at {raw_images}")

    leaf_map = load_leaf_map(leaf_map_path)
    train_names = _load_split(splits_dir / "color_train.txt")
    test_names = _load_split(splits_dir / "color_test.txt")

    class_dirs = sorted(d for d in raw_images.iterdir() if d.is_dir())
    class_to_label = {d.name: i for i, d in enumerate(class_dirs)}

    rows: list[dict] = []
    for class_dir in class_dirs:
        crop, disease = parse_class_name(class_dir.name)
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        if max_per_class is not None:
            files = files[:max_per_class]

        for path in files:
            key = _leaf_key(path)
            entry = leaf_map.get(key)
            if entry and ":::" in entry:
                # "<class>:::<group id>" - namespace the id by class so ids from
                # different classes can never collide.
                cls, gid = entry.rsplit(":::", 1)
                leaf_group = f"{cls}:::{gid}"
            else:
                # Unmapped image: it is its own group, which is the safe default.
                leaf_group = f"{class_dir.name}:::{key}"

            if path.name in train_names:
                split = "train"
            elif path.name in test_names:
                split = "test"
            else:
                split = "unassigned"

            rows.append(
                {
                    "image_path": str(path),
                    "filename": path.name,
                    "class_name": class_dir.name,
                    "crop": crop,
                    "disease": disease,
                    "is_healthy": disease.lower() in HEALTHY_TOKENS,
                    "leaf_group": leaf_group,
                    "split": split,
                    "label": class_to_label[class_dir.name],
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["class_name", "filename"]).reset_index(drop=True)


def class_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class summary: image count, leaf count, and split breakdown."""
    out = (
        df.groupby(["class_name", "crop", "disease", "is_healthy"])
        .agg(
            n_images=("image_path", "size"),
            n_leaves=("leaf_group", "nunique"),
            n_train=("split", lambda s: int((s == "train").sum())),
            n_test=("split", lambda s: int((s == "test").sum())),
        )
        .reset_index()
        .sort_values("n_images", ascending=False)
    )
    out["imgs_per_leaf"] = (out.n_images / out.n_leaves).round(2)
    return out
