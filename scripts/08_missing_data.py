"""Stage 8 - When does the graph actually earn its place?

On the full dataset a frozen transformer already identifies the disease with
~98% accuracy, so message passing has almost no headroom and mainly dilutes a
strong per-node signal. That is an honest result, and it is reported as such.

But the operational case for a graph is different: **not every farm submits a
photograph every day**. A farm with no image has no visual evidence at all, and
the only information available about it is its own weather plus what its
neighbours are seeing. That is precisely what a GNN is for.

This experiment masks the vision stream for a growing fraction of farms and
measures GraphSAGE against the identical no-graph MLP. If the graph is worth
having, the gap should open as observations become sparse.

Run:  python scripts/08_missing_data.py
Out:  outputs/reports/stage8_missing_data.csv
"""
from __future__ import annotations

import argparse
import copy
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


def mask_vision(splits, fraction: float, seed: int):
    """Zero the vision stream for a random fraction of nodes in every split.

    A masked node is a farm that did not submit a photograph that day. Its
    climate and metadata remain, so the model still knows where and when it is.
    """
    if fraction <= 0:
        return splits
    out = {}
    for i, (name, st) in enumerate(splits.items()):
        g = torch.Generator(device="cpu").manual_seed(seed + i)
        keep = (torch.rand(st.n, generator=g) >= fraction).float().unsqueeze(1)
        clone = copy.copy(st)
        clone.vision = st.vision * keep.to(st.vision.device)
        out[name] = clone
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 0.9])
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    device = Device.auto(cfg.training.amp).torch()
    backbone = args.backbone or cfg.backbones.primary
    reports = Path(cfg.paths.reports)

    print("=" * 78)
    print("STAGE 8  |  Does the graph help when observations are missing?")
    print("=" * 78)

    splits, meta = prepare_splits(cfg, backbone, "leaf_disjoint", device, verbose=False)
    print(f"train edges: {splits['train'].edge_index.shape[1]:,}")

    rows = []
    for frac in args.fractions:
        data = mask_vision(splits, frac, cfg.project.seed)
        print(f"\n--- {frac:.0%} of farms have no photograph ---")
        for conv in ("sage", "none"):
            set_seed(cfg.project.seed)
            model = CropDiseaseForecastNet(
                vision_dim=meta["vision_dim"], climate_dim=meta["climate_dim"],
                meta_dim=meta["meta_dim"], num_classes=meta["num_classes"],
                horizons=tuple(meta["horizons"]),
                fusion_dim=cfg.model.fusion.hidden_dim,
                gnn_conv=conv, gnn_hidden=cfg.model.gnn.hidden_dim,
                gnn_layers=cfg.model.gnn.num_layers,
                gnn_heads=cfg.model.gnn.heads,
            )
            tcfg = TrainConfig(
                epochs=args.epochs, lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                lambda_physics=cfg.physics.lambda_physics,
                physics_weights=PhysicsWeights(**dict(cfg.physics.weights)),
                log_every=10 ** 9,
            )
            model, _ = train_model(model, data["train"], data["val"], tcfg,
                                   device, use_physics=True, verbose=False)
            test = evaluate(model, data["test"])
            rows.append({
                "missing_fraction": frac,
                "encoder": "GraphSAGE" if conv == "sage" else "MLP (no graph)",
                "test_acc": test["acc"], "test_macro_f1": test["macro_f1"],
                "risk_r2_mean": test["risk_r2_mean"],
                "band_acc_mean": test["band_acc_mean"],
            })
            print(f"    {rows[-1]['encoder']:16s} acc {test['acc']:.4f}  "
                  f"F1 {test['macro_f1']:.4f}  riskR2 {test['risk_r2_mean']:+.4f}")
            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(reports / "stage8_missing_data.csv", index=False)

    # The headline: graph minus no-graph, at each level of sparsity.
    pivot = df.pivot(index="missing_fraction", columns="encoder",
                     values="test_macro_f1")
    pivot["graph_gain_F1"] = pivot["GraphSAGE"] - pivot["MLP (no graph)"]
    pivot_r2 = df.pivot(index="missing_fraction", columns="encoder",
                        values="risk_r2_mean")
    pivot["graph_gain_riskR2"] = pivot_r2["GraphSAGE"] - pivot_r2["MLP (no graph)"]

    print("\n" + "=" * 78)
    print("GRAPH VALUE vs OBSERVATION SPARSITY")
    print("=" * 78)
    print(pivot.round(4).to_string())
    print(f"\nsaved -> {reports/'stage8_missing_data.csv'}")


if __name__ == "__main__":
    main()
