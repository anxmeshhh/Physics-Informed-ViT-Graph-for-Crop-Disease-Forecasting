"""Stage 3 - Vision transformer benchmark.

Compares DINOv2, ViT-B/16, Swin-T and DeiT-S under an identical downstream
pipeline, so the only thing that varies is the frozen visual representation.

Two protocols are reported for each backbone:

*linear probe*  - a single linear layer on the frozen embedding. The standard
                  measure of raw representation quality.
*full pipeline* - the complete fusion + GNN + physics-informed model.

Both are evaluated on the leaf-disjoint split, so no photograph of a test leaf
was ever seen in training.

Run:  python scripts/03_benchmark_backbones.py
Out:  outputs/reports/stage3_backbone_benchmark.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cropforecast                                                        # noqa: E402,F401
from cropforecast.config import Device, ensure_dirs, load_config, set_seed  # noqa: E402
from cropforecast.models.backbones import SPECS                             # noqa: E402
from cropforecast.models.full_model import CropDiseaseForecastNet           # noqa: E402
from cropforecast.train.prepare import prepare_splits                       # noqa: E402
from cropforecast.train.trainer import (                                    # noqa: E402
    PhysicsWeights, TrainConfig, evaluate, train_model,
)


def linear_probe(splits, num_classes: int, device, epochs: int = 400,
                 lr: float = 1e-3) -> dict[str, float]:
    """Train a single linear layer on the frozen embedding."""
    tr, te = splits["train"], splits["test"]
    probe = torch.nn.Linear(tr.vision.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(probe(tr.vision), tr.labels).backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(te.vision).argmax(1)
    acc = float((pred == te.labels).float().mean())

    f1s = []
    for c in range(num_classes):
        tp = float(((pred == c) & (te.labels == c)).sum())
        fp = float(((pred == c) & (te.labels != c)).sum())
        fn = float(((pred != c) & (te.labels == c)).sum())
        if tp + fp + fn > 0:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"probe_acc": acc, "probe_macro_f1": float(np.mean(f1s))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["dinov2", "vit", "swin", "deit"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--split", default="leaf_disjoint")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    dev = Device.auto(cfg.training.amp)
    device = dev.torch()
    reports = Path(cfg.paths.reports)

    print("=" * 78)
    print(f"STAGE 3  |  Vision transformer benchmark  ({args.split} split)")
    print("=" * 78)

    rows = []
    for key in args.backbones:
        set_seed(cfg.project.seed)
        spec = SPECS[key]
        print(f"\n--- {key}  ({spec.hf_id}) ---")

        splits, meta = prepare_splits(cfg, key, split_mode=args.split, device=device)

        t0 = time.time()
        probe = linear_probe(splits, meta["num_classes"], device)
        print(f"    linear probe  acc {probe['probe_acc']:.4f}  "
              f"F1 {probe['probe_macro_f1']:.4f}  ({time.time()-t0:.0f}s)")

        model = CropDiseaseForecastNet(
            vision_dim=meta["vision_dim"],
            climate_dim=meta["climate_dim"],
            meta_dim=meta["meta_dim"],
            num_classes=meta["num_classes"],
            horizons=tuple(meta["horizons"]),
            fusion_dim=cfg.model.fusion.hidden_dim,
            gnn_conv=cfg.model.gnn.conv,
            gnn_hidden=cfg.model.gnn.hidden_dim,
            gnn_layers=cfg.model.gnn.num_layers,
            gnn_heads=cfg.model.gnn.heads,
            dropout=cfg.model.fusion.dropout,
            gnn_dropout=cfg.model.gnn.dropout,
        )
        tcfg = TrainConfig(
            epochs=args.epochs, lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            lambda_physics=cfg.physics.lambda_physics,
            physics_weights=PhysicsWeights(**dict(cfg.physics.weights)),
        )
        model, _ = train_model(model, splits["train"], splits["val"], tcfg,
                               device, use_physics=True)
        test = evaluate(model, splits["test"])

        rows.append({
            "backbone": key,
            "checkpoint": spec.hf_id,
            "family": spec.family,
            "dim": spec.dim,
            "pool": spec.pool,
            **probe,
            "full_acc": test["acc"],
            "full_macro_f1": test["macro_f1"],
            "risk_r2_mean": test["risk_r2_mean"],
            "band_acc_mean": test["band_acc_mean"],
            "params_trainable_M": model.num_trainable() / 1e6,
        })
        print(f"    full pipeline acc {test['acc']:.4f}  F1 {test['macro_f1']:.4f}  "
              f"riskR2 {test['risk_r2_mean']:+.4f}")

        del model, splits
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("full_macro_f1", ascending=False)
    out = reports / "stage3_backbone_benchmark.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("BACKBONE BENCHMARK  (leaf-disjoint test set)")
    print("=" * 78)
    print(df[["backbone", "dim", "probe_acc", "probe_macro_f1",
              "full_acc", "full_macro_f1", "risk_r2_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
