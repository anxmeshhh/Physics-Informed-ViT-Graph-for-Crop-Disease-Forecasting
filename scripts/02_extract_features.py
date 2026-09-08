"""Stage 2 - Extract frozen vision embeddings for every backbone.

Each JPEG is decoded and resized once, then pushed through all four transformers
with their own normalisation statistics. Embeddings are cached to
``data/processed/emb_<backbone>.npy`` and reused by every later stage.

Run:  python scripts/02_extract_features.py [--backbones dinov2 vit swin deit]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cropforecast                                                       # noqa: E402,F401
from cropforecast.config import Device, ensure_dirs, load_config, set_seed  # noqa: E402
from cropforecast.data.dataset import RawImageDataset, normalise          # noqa: E402
from cropforecast.models.backbones import SPECS, load_backbone            # noqa: E402
from transformers import AutoImageProcessor                               # noqa: E402


def norm_stats(hf_id: str, device: torch.device):
    """Per-checkpoint normalisation constants, shaped for broadcasting."""
    proc = AutoImageProcessor.from_pretrained(hf_id)
    mean = torch.tensor(getattr(proc, "image_mean", [0.485, 0.456, 0.406]),
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor(getattr(proc, "image_std", [0.229, 0.224, 0.225]),
                       device=device).view(1, 3, 1, 1)
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=sorted(SPECS))
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.project.seed)
    dev = Device.auto(cfg.training.amp)
    device = dev.torch()
    processed = Path(cfg.paths.processed)

    # Index EVERY PlantVillage image, not just the ones the current grounding
    # happens to use. The assignment in stage 1 is stochastic and drops rows
    # whose forecast horizon runs past the archive, so an observations-derived
    # cache silently goes stale the moment stage 1 is re-run with different
    # settings. Keying the cache to the full image set makes it reusable.
    from cropforecast.data.plantvillage import build_index                  # noqa: E402

    index = build_index(cfg.paths.raw_images, cfg.paths.splits, cfg.paths.leaf_map,
                        max_per_class=cfg.data.max_images_per_class)
    if args.limit:
        index = index.head(args.limit)
    paths = index.image_path.tolist()

    batch_size = args.batch_size or cfg.backbones.batch_size
    ds = RawImageDataset(paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=cfg.backbones.num_workers, pin_memory=True)

    print("=" * 78)
    print(f"STAGE 2  |  Frozen embedding extraction on {device.type.upper()}")
    print(f"          {len(paths):,} images | batch {batch_size} | "
          f"backbones: {', '.join(args.backbones)}")
    print("=" * 78)

    # Load every backbone once; they are small enough to co-reside in 6 GB.
    models, stats = {}, {}
    for key in args.backbones:
        models[key] = load_backbone(key, frozen=True).to(device).eval()
        stats[key] = norm_stats(SPECS[key].hf_id, device)
        print(f"  loaded {key:8s} dim={models[key].dim}")

    out = {k: np.zeros((len(paths), models[k].dim), dtype=np.float32)
           for k in args.backbones}

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for batch_u8, idx in loader:
            batch_u8 = batch_u8.to(device, non_blocking=True)
            for key in args.backbones:
                mean, std = stats[key]
                x = normalise(batch_u8, mean, std)
                with torch.autocast("cuda", enabled=dev.amp):
                    emb = models[key](x)["pooled"]
                out[key][idx.numpy()] = emb.float().cpu().numpy()
            done += len(idx)
            if done % (batch_size * 20) < batch_size:
                rate = done / (time.time() - t0)
                eta = (len(paths) - done) / max(rate, 1e-9)
                print(f"    {done:6,}/{len(paths):,}  {rate:6.1f} img/s  "
                      f"ETA {eta/60:5.1f} min", flush=True)

    print(f"\n  extraction took {(time.time()-t0)/60:.1f} min")
    for key in args.backbones:
        path = processed / f"emb_{key}.npy"
        np.save(path, out[key])
        arr = out[key]
        print(f"  saved {path.name:18s} shape={arr.shape} "
              f"mean={arr.mean():+.4f} std={arr.std():.4f} "
              f"({arr.nbytes/1e6:.0f} MB)")

    # Row order is the contract between the cache and every later stage.
    # Stages join on image_path, never on position.
    index[["image_path", "class_name", "crop", "leaf_group"]].to_parquet(
        processed / "embedding_row_order.parquet", index=False)
    print("\n  row order saved -> embedding_row_order.parquet")


if __name__ == "__main__":
    main()
