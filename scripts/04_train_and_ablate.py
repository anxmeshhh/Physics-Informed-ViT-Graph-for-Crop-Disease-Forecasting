"""Stage 4 - Train the final model and run the ablation study.

Answers the questions a reviewer will actually ask:

* Does the graph earn its place, or is an MLP on the same features as good?
* Does the physics loss help, or is it decoration?
* How much does each input stream contribute?
* Which graph encoder is best - GraphSAGE, GCN or GAT?
* How much accuracy does leaf-level leakage manufacture?

Run:  python scripts/04_train_and_ablate.py
Out:  outputs/checkpoints/final_model.pt
      outputs/reports/stage4_ablations.csv
      outputs/reports/stage4_history.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cropforecast                                                        # noqa: E402,F401
from cropforecast.config import Device, ensure_dirs, load_config, set_seed  # noqa: E402
from cropforecast.models.full_model import CropDiseaseForecastNet           # noqa: E402
from cropforecast.train.prepare import prepare_splits                       # noqa: E402
from cropforecast.train.trainer import (                                    # noqa: E402
    PhysicsWeights, TrainConfig, evaluate, train_model,
)


def build(cfg, meta, conv: str) -> CropDiseaseForecastNet:
    return CropDiseaseForecastNet(
        vision_dim=meta["vision_dim"],
        climate_dim=meta["climate_dim"],
        meta_dim=meta["meta_dim"],
        num_classes=meta["num_classes"],
        horizons=tuple(meta["horizons"]),
        fusion_dim=cfg.model.fusion.hidden_dim,
        gnn_conv=conv,
        gnn_hidden=cfg.model.gnn.hidden_dim,
        gnn_layers=cfg.model.gnn.num_layers,
        gnn_heads=cfg.model.gnn.heads,
        dropout=cfg.model.fusion.dropout,
        gnn_dropout=cfg.model.gnn.dropout,
    )


def mask_stream(splits, stream: str):
    """Return a copy of the splits with one input stream zeroed out."""
    import copy as _copy
    out = {}
    for name, st in splits.items():
        clone = _copy.copy(st)
        setattr(clone, stream, torch.zeros_like(getattr(st, stream)))
        out[name] = clone
    return out


def run(cfg, splits, meta, device, name, conv="sage", use_physics=True,
        epochs=300, stream_mask=None):
    """Train one configuration and return its test metrics."""
    set_seed(cfg.project.seed)
    data = mask_stream(splits, stream_mask) if stream_mask else splits

    model = build(cfg, meta, conv)
    tcfg = TrainConfig(
        epochs=epochs, lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        lambda_physics=cfg.physics.lambda_physics if use_physics else 0.0,
        physics_weights=PhysicsWeights(**dict(cfg.physics.weights)),
    )
    print(f"\n--- {name} ---")
    model, history = train_model(model, data["train"], data["val"], tcfg,
                                 device, use_physics=use_physics)
    test = evaluate(model, data["test"])
    print(f"    TEST acc {test['acc']:.4f}  F1 {test['macro_f1']:.4f}  "
          f"riskR2 {test['risk_r2_mean']:+.4f}  band {test['band_acc_mean']:.4f}")
    return model, history, {
        "config": name, "encoder": conv, "physics": use_physics,
        "masked_stream": stream_mask or "-",
        "test_acc": test["acc"], "test_macro_f1": test["macro_f1"],
        "risk_r2_mean": test["risk_r2_mean"], "band_acc_mean": test["band_acc_mean"],
        **{k: v for k, v in test.items() if k.startswith(("risk_r2_h", "risk_mae_h"))},
        "trainable_params_M": model.num_trainable() / 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--with-gat", action="store_true",
                    help="include the GAT encoder (slow: needs checkpointing)")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    dev = Device.auto(cfg.training.amp)
    device = dev.torch()
    backbone = args.backbone or cfg.backbones.primary
    reports, ckpts = Path(cfg.paths.reports), Path(cfg.paths.checkpoints)

    print("=" * 78)
    print(f"STAGE 4  |  Final model + ablations  (backbone={backbone})")
    print("=" * 78)

    print("\nPreparing leaf-disjoint splits ...")
    splits, meta = prepare_splits(cfg, backbone, "leaf_disjoint", device)

    results, best_model, best_history = [], None, None

    # --- The full model ----------------------------------------------------
    model, history, row = run(cfg, splits, meta, device,
                              "FULL (ViT + climate + graph + physics)",
                              conv=cfg.model.gnn.conv, use_physics=True,
                              epochs=args.epochs)
    results.append(row)
    best_model, best_history = model, history

    # --- Component ablations ----------------------------------------------
    for name, kwargs in [
        ("no physics loss",        dict(conv=cfg.model.gnn.conv, use_physics=False)),
        ("no graph (MLP encoder)", dict(conv="none", use_physics=True)),
        ("no climate stream",      dict(conv=cfg.model.gnn.conv, use_physics=True,
                                        stream_mask="climate")),
        ("no vision stream",       dict(conv=cfg.model.gnn.conv, use_physics=True,
                                        stream_mask="vision")),
    ]:
        _, _, row = run(cfg, splits, meta, device, name, epochs=args.epochs, **kwargs)
        results.append(row)

    # --- Graph encoder comparison -----------------------------------------
    # GAT is gated behind a flag. It needs gradient checkpointing to fit half a
    # million edges into 6 GB, which makes it several times slower than the
    # others for what is one row of a comparison table - and it otherwise blocks
    # the checkpoint that the dashboard and the explainability stage depend on.
    encoders = ["gcn"] + (["gat"] if args.with_gat else [])
    for conv in encoders:
        if conv == cfg.model.gnn.conv:
            continue
        _, _, row = run(cfg, splits, meta, device, f"encoder = {conv.upper()}",
                        conv=conv, use_physics=True, epochs=args.epochs)
        results.append(row)

    # --- Leakage demonstration --------------------------------------------
    print("\nPreparing RANDOM (leaky) splits for comparison ...")
    leaky_splits, leaky_meta = prepare_splits(cfg, backbone, "random", device)
    _, _, row = run(cfg, leaky_splits, leaky_meta, device,
                    "FULL on RANDOM split (leaky)", conv=cfg.model.gnn.conv,
                    use_physics=True, epochs=args.epochs)
    row["masked_stream"] = "random-split"
    results.append(row)

    # --- Save --------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(reports / "stage4_ablations.csv", index=False)
    pd.DataFrame(best_history).to_csv(reports / "stage4_history.csv", index=False)

    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "meta": {k: v for k, v in meta.items()
                     if k not in ("feature_space", "obs")},
            "config": {
                "backbone": backbone,
                "gnn_conv": cfg.model.gnn.conv,
                "fusion_dim": cfg.model.fusion.hidden_dim,
                "gnn_hidden": cfg.model.gnn.hidden_dim,
                "gnn_layers": cfg.model.gnn.num_layers,
                "gnn_heads": cfg.model.gnn.heads,
            },
        },
        ckpts / "final_model.pt",
    )
    import pickle
    with open(ckpts / "feature_space.pkl", "wb") as fh:
        pickle.dump(meta["feature_space"], fh)
    (ckpts / "class_names.json").write_text(
        json.dumps(
            meta["obs"][["label", "class_name"]]
            .drop_duplicates().sort_values("label").class_name.tolist()
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("ABLATION STUDY  (leaf-disjoint test set unless noted)")
    print("=" * 78)
    print(df[["config", "test_acc", "test_macro_f1", "risk_r2_mean",
              "band_acc_mean"]].to_string(index=False,
                                          float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved -> {reports/'stage4_ablations.csv'}")
    print(f"saved -> {ckpts/'final_model.pt'}")


if __name__ == "__main__":
    main()
