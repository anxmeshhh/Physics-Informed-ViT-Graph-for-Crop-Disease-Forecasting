"""Stage 5 - Explainability (Block 6 of the architecture).

Produces every explainability artefact the architecture diagram names:

  * Grad-CAM and attention rollout over leaf images
  * Integrated Gradients over the climate variables
  * SHAP global importance over the climate block
  * Graph attribution: which neighbouring farms drive a forecast

Run:  python scripts/05_explain.py
Out:  outputs/figures/explain_*.png
      outputs/reports/stage5_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402
import pandas as pd                                                        # noqa: E402
import torch                                                               # noqa: E402
from PIL import Image                                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cropforecast                                                        # noqa: E402,F401
from cropforecast.config import Device, ensure_dirs, load_config, set_seed  # noqa: E402
from cropforecast.explain.attribution import (                              # noqa: E402
    explain_graph, integrated_gradients, shap_climate,
)
from cropforecast.explain.vision import attention_rollout, grad_cam, overlay  # noqa: E402
from cropforecast.models.backbones import SPECS, load_backbone              # noqa: E402
from cropforecast.models.full_model import CropDiseaseForecastNet           # noqa: E402
from cropforecast.train.prepare import prepare_splits                       # noqa: E402


def load_trained(cfg, device):
    ckpt = Path(cfg.paths.checkpoints) / "final_model.pt"
    if not ckpt.exists():
        raise SystemExit("No trained model. Run scripts/04_train_and_ablate.py first.")
    blob = torch.load(ckpt, map_location=device)
    m, c = blob["meta"], blob["config"]
    model = CropDiseaseForecastNet(
        vision_dim=m["vision_dim"], climate_dim=m["climate_dim"],
        meta_dim=m["meta_dim"], num_classes=m["num_classes"],
        horizons=tuple(m["horizons"]), fusion_dim=c["fusion_dim"],
        gnn_conv=c["gnn_conv"], gnn_hidden=c["gnn_hidden"],
        gnn_layers=c["gnn_layers"], gnn_heads=c["gnn_heads"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, m, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.project.seed)
    device = Device.auto(cfg.training.amp).torch()
    figures, reports = Path(cfg.paths.figures), Path(cfg.paths.reports)

    print("=" * 78)
    print("STAGE 5  |  Explainability")
    print("=" * 78)

    model, meta, model_cfg = load_trained(cfg, device)
    backbone_key = model_cfg["backbone"]
    splits, prep = prepare_splits(cfg, backbone_key, "leaf_disjoint", device, verbose=False)
    obs = prep["obs"]
    test_obs = obs[obs.split == "test"].reset_index(drop=True)

    # ---- 1. Vision explanations ------------------------------------------
    print("\n[1/4] Grad-CAM and attention rollout on leaf images ...")
    backbone = load_backbone(backbone_key, frozen=True, output_attentions=True).to(device).eval()

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(SPECS[backbone_key].hf_id)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)

    # Pick visually distinct diseased examples rather than random rows.
    wanted = ["Tomato___Late_blight", "Apple___Apple_scab", "Grape___Black_rot",
              "Potato___Early_blight", "Squash___Powdery_mildew",
              "Corn_(maize)___Common_rust_"]
    picks = []
    for cls in wanted[: args.n_images]:
        sub = test_obs[test_obs.class_name == cls]
        if len(sub):
            picks.append(sub.iloc[0])
    if not picks:
        picks = [test_obs.iloc[i] for i in range(min(args.n_images, len(test_obs)))]

    fig, axes = plt.subplots(3, len(picks), figsize=(3.1 * len(picks), 9.4))
    if len(picks) == 1:
        axes = axes.reshape(3, 1)

    class_names = sorted(obs.class_name.unique())
    for col, row in enumerate(picks):
        img = Image.open(row.image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device).float() / 255.0
        x = (x - mean) / std

        roll = attention_rollout(backbone, x)
        cam, used = grad_cam(backbone, _ProxyHead(model, backbone.dim, device), x)

        axes[0, col].imshow(arr)
        axes[0, col].set_title(row.class_name.replace("___", "\n").replace("_", " "),
                               fontsize=8)
        axes[1, col].imshow(overlay(arr, roll))
        axes[2, col].imshow(overlay(arr, cam))
        for r in range(3):
            axes[r, col].axis("off")

    axes[0, 0].set_ylabel("input")
    for r, name in enumerate(["Input leaf", "Attention rollout", "Grad-CAM"]):
        axes[r, 0].text(-0.12, 0.5, name, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle(f"Visual explanations - {backbone_key}", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures / "explain_vision.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved {figures/'explain_vision.png'}")

    # ---- 2. Integrated gradients over climate -----------------------------
    print("\n[2/4] Integrated Gradients over climate variables ...")
    te = splits["test"]
    n = min(512, te.n)
    ig = integrated_gradients(
        model, te.vision[:n], te.climate[:n].clone().requires_grad_(True),
        te.meta[:n], prep["climate_cols"], horizon=0,
    )
    ig.to_csv(reports / "stage5_integrated_gradients.csv", index=False)
    print(f"      convergence delta {ig.attrs.get('convergence_delta', float('nan')):.4f}")
    print(ig.head(10)[["feature", "attribution", "abs_attribution"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    top = ig.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#f43f5e" if v < 0 else "#22c55e" for v in top.attribution]
    ax.barh(top.feature, top.attribution, color=colours)
    ax.set_xlabel("Attribution to 1-day risk forecast")
    ax.set_title("Integrated Gradients: which weather drives the forecast")
    ax.axvline(0, color="#888", lw=.8)
    fig.tight_layout()
    fig.savefig(figures / "explain_integrated_gradients.png", dpi=150)
    plt.close(fig)

    # ---- 3. SHAP ----------------------------------------------------------
    print("\n[3/4] SHAP global importance ...")
    try:
        sh = shap_climate(model, te.vision, te.climate, te.meta,
                          prep["climate_cols"], horizon=0)
        sh.to_csv(reports / "stage5_shap.csv", index=False)
        print(sh.head(10).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

        top = sh.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top.feature, top.mean_abs_shap, color="#38bdf8")
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title("SHAP: global climate feature importance")
        fig.tight_layout()
        fig.savefig(figures / "explain_shap.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"      SHAP skipped: {exc}")

    # ---- 4. Graph attribution --------------------------------------------
    print("\n[4/4] Graph attribution ...")
    deg = torch.bincount(te.edge_index[1], minlength=te.n)
    node = int(deg.argmax())
    neigh = explain_graph(model, te, node, test_obs, horizon=0, top_k=10)
    neigh.to_csv(reports / "stage5_graph_attribution.csv", index=False)
    target = test_obs.iloc[node]
    print(f"      target node {node}: {target.site_id} {str(target.date)[:10]} "
          f"({target.class_name})")
    if len(neigh):
        cols = [c for c in ["site_id", "date", "importance_pct", "dist_km_norm",
                            "is_downwind"] if c in neigh.columns]
        print(neigh[cols].to_string(index=False))

    print("\nAll explainability artefacts written to outputs/")


class _ProxyHead(torch.nn.Module):
    """Maps a backbone embedding to class logits through the trained stack.

    Grad-CAM needs logits as a function of the pooled embedding alone, so the
    climate and metadata streams are pinned to their dataset means (zero in
    standardised space) and only the vision path carries gradient.
    """

    def __init__(self, model, vision_dim: int, device):
        super().__init__()
        self.model = model
        self.device = device
        self.climate_dim = model.fusion.climate_enc.net[0].in_features
        self.meta_dim = model.fusion.meta_enc.net[0].in_features

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        b = pooled.shape[0]
        climate = torch.zeros(b, self.climate_dim, device=pooled.device)
        meta = torch.zeros(b, self.meta_dim, device=pooled.device)
        empty = torch.zeros((2, 0), dtype=torch.long, device=pooled.device)
        return self.model(pooled, climate, meta, empty, None)["class_logits"]


if __name__ == "__main__":
    main()
