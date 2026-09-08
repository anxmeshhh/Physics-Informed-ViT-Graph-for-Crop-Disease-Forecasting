"""Stage 7 - Publication-quality figures for the report and slides.

Run:  python scripts/07_figures.py
Out:  outputs/figures/fig_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402
import pandas as pd                                                        # noqa: E402
import torch                                                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cropforecast                                                        # noqa: E402,F401
from cropforecast.config import Device, ensure_dirs, load_config, set_seed  # noqa: E402
from cropforecast.data.farms import to_frame                                # noqa: E402
from cropforecast.graph.build import site_distance_matrix                   # noqa: E402
from cropforecast.models.full_model import CropDiseaseForecastNet           # noqa: E402
from cropforecast.physics.epidemiology import (                             # noqa: E402
    PROFILES, temperature_response,
)
from cropforecast.train.prepare import prepare_splits                       # noqa: E402
from cropforecast.train.trainer import evaluate                             # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
GREEN, BLUE, AMBER, ROSE = "#22c55e", "#38bdf8", "#fbbf24", "#f43f5e"


def fig_backbones(reports: Path, figures: Path) -> None:
    path = reports / "stage3_backbone_benchmark.csv"
    if not path.exists():
        print("  skip backbones (run stage 3)")
        return
    df = pd.read_csv(path).sort_values("full_macro_f1")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(df))
    ax[0].barh(y - .2, df.probe_macro_f1, .38, label="linear probe", color=BLUE)
    ax[0].barh(y + .2, df.full_macro_f1, .38, label="full pipeline", color=GREEN)
    ax[0].set_yticks(y, df.backbone)
    ax[0].set_xlabel("macro-F1 (38 classes)")
    ax[0].set_title("Disease classification")
    ax[0].legend()

    ax[1].barh(y, df.risk_r2_mean, .5, color=AMBER)
    ax[1].set_yticks(y, df.backbone)
    ax[1].set_xlabel("mean risk R² across horizons")
    ax[1].set_title("Forecast skill")

    fig.suptitle("Vision transformer benchmark (leaf-disjoint test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / "fig_backbones.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  fig_backbones.png")


def fig_ablations(reports: Path, figures: Path) -> None:
    path = reports / "stage4_ablations.csv"
    if not path.exists():
        print("  skip ablations (run stage 4)")
        return
    df = pd.read_csv(path)
    leaky = df[df.masked_stream == "random-split"]
    main = df[df.masked_stream != "random-split"].sort_values("test_macro_f1")

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    y = np.arange(len(main))
    colours = [GREEN if c.startswith("FULL (") else BLUE for c in main.config]
    ax[0].barh(y, main.test_macro_f1, color=colours)
    ax[0].set_yticks(y, main.config, fontsize=9)
    ax[0].set_xlabel("macro-F1")
    ax[0].set_title("Ablation: disease classification")

    ax[1].barh(y, main.risk_r2_mean, color=colours)
    ax[1].set_yticks(y, main.config, fontsize=9)
    ax[1].set_xlabel("mean risk R²")
    ax[1].set_title("Ablation: forecast skill")

    if len(leaky):
        full = main[main.config.str.startswith("FULL (")]
        if len(full):
            gap = leaky.test_macro_f1.iloc[0] - full.test_macro_f1.iloc[0]
            fig.suptitle(
                "Component ablations  —  random (leaky) split inflates macro-F1 by "
                f"{gap:+.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / "fig_ablations.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  fig_ablations.png")


def fig_horizon(reports: Path, figures: Path, cfg) -> None:
    path = reports / "stage4_ablations.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    full = df[df.config.str.startswith("FULL (")]
    if not len(full):
        return
    horizons = list(cfg.model.heads.horizons)
    r2 = [full[f"risk_r2_h{i}"].iloc[0] for i in range(len(horizons))
          if f"risk_r2_h{i}" in full.columns]
    mae = [full[f"risk_mae_h{i}"].iloc[0] for i in range(len(horizons))
           if f"risk_mae_h{i}" in full.columns]
    if not r2:
        return

    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax1.plot(horizons[:len(r2)], r2, "o-", color=GREEN, lw=2, label="R²")
    ax1.set_xlabel("forecast horizon (days ahead)")
    ax1.set_ylabel("R²", color=GREEN)
    ax2 = ax1.twinx()
    ax2.plot(horizons[:len(mae)], mae, "s--", color=ROSE, lw=2, label="MAE")
    ax2.set_ylabel("MAE", color=ROSE)
    ax2.grid(False)
    ax1.set_title("Forecast skill decays with horizon, as it must")
    fig.tight_layout()
    fig.savefig(figures / "fig_horizon_skill.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  fig_horizon_skill.png")


def fig_confusion(cfg, figures: Path, device) -> None:
    ckpt = Path(cfg.paths.checkpoints) / "final_model.pt"
    if not ckpt.exists():
        print("  skip confusion (run stage 4)")
        return
    blob = torch.load(ckpt, map_location=device)
    m, c = blob["meta"], blob["config"]
    model = CropDiseaseForecastNet(
        vision_dim=m["vision_dim"], climate_dim=m["climate_dim"],
        meta_dim=m["meta_dim"], num_classes=m["num_classes"],
        horizons=tuple(m["horizons"]), fusion_dim=c["fusion_dim"],
        gnn_conv=c["gnn_conv"], gnn_hidden=c["gnn_hidden"],
        gnn_layers=c["gnn_layers"], gnn_heads=c["gnn_heads"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    splits, prep = prepare_splits(cfg, c["backbone"], "leaf_disjoint", device,
                                  verbose=False)
    te = splits["test"]
    with torch.no_grad():
        pred = model(te.vision, te.climate, te.meta,
                     te.edge_index, te.edge_attr)["class_logits"].argmax(1)

    obs = prep["obs"]
    names = obs[["label", "class_name"]].drop_duplicates().sort_values("label")
    labels = names.class_name.tolist()
    n = len(labels)

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(te.labels.cpu().numpy(), pred.cpu().numpy()):
        cm[t, p] += 1
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(13, 11.5))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    acc = float((pred == te.labels).float().mean())
    ax.set_title(f"Confusion matrix — leaf-disjoint test set (accuracy {acc:.3f})")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.8, label="row-normalised")
    fig.tight_layout()
    fig.savefig(figures / "fig_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_confusion_matrix.png (test accuracy {acc:.4f})")


def fig_pathogen_response(figures: Path) -> None:
    t = np.linspace(0, 42, 400)
    show = [
        ("Potato___Late_blight", "Late blight", BLUE),
        ("Apple___Apple_scab", "Apple scab", "#8b5cf6"),
        ("Squash___Powdery_mildew", "Powdery mildew", GREEN),
        ("Tomato___Bacterial_spot", "Bacterial spot", AMBER),
        ("Tomato___Spider_mites Two-spotted_spider_mite", "Spider mites", ROSE),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for key, label, colour in show:
        ax.plot(t, temperature_response(t, PROFILES[key]), lw=2.2,
                label=f"{label} (opt {PROFILES[key].t_opt:.0f}°C)", color=colour)
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("relative development rate")
    ax.set_title("Analytis beta model: each pathogen has its own cardinal temperatures")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "fig_pathogen_response.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  fig_pathogen_response.png")


def fig_network(figures: Path, cfg) -> None:
    sites = to_frame()
    d, _, ids = site_distance_matrix(sites)
    k, radius, minnb = cfg.graph.k_neighbours, cfg.graph.radius_km, cfg.graph.min_neighbours

    fig, ax = plt.subplots(figsize=(9, 10))
    for i in range(len(ids)):
        order = [j for j in d[i].argsort() if j != i]
        within = [j for j in order if d[i, j] <= radius][:k]
        if len(within) < minnb:
            within = order[:minnb]
        for j in within:
            ax.plot([sites.lon[i], sites.lon[j]], [sites.lat[i], sites.lat[j]],
                    color=GREEN, alpha=.28, lw=.9, zorder=1)
    ax.scatter(sites.lon, sites.lat, s=80, c=ROSE, zorder=2,
               edgecolor="white", linewidth=1.2)
    for _, r in sites.iterrows():
        ax.annotate(r["name"], (r.lon, r.lat), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Farm network: {len(sites)} districts, "
                 f"KNN(k={k}) + radius({radius:.0f} km)")
    fig.tight_layout()
    fig.savefig(figures / "fig_farm_network.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  fig_farm_network.png")


def fig_climate_separation(cfg, figures: Path) -> None:
    obs = pd.read_parquet(Path(cfg.paths.processed) / "observations.parquet")
    agg = obs.groupby("class_name").agg(
        temp=("temperature_2m_mean", "mean"),
        wet=("leaf_wetness_hours", "mean"),
        n=("image_path", "size"),
    ).reset_index()
    agg["healthy"] = agg.class_name.str.endswith("___healthy")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for flag, colour, label in [(False, ROSE, "diseased"), (True, GREEN, "healthy")]:
        s = agg[agg.healthy == flag]
        ax.scatter(s.temp, s.wet, s=np.sqrt(s.n) * 5, c=colour, alpha=.65,
                   edgecolor="white", label=label)
    for _, r in agg.iterrows():
        ax.annotate(r.class_name.split("___")[1][:18].replace("_", " "),
                    (r.temp, r.wet), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("mean temperature at observation (°C)")
    ax.set_ylabel("mean leaf wetness (hours)")
    ax.set_title("Each disease occupies its own climate niche\n"
                 "(marker area ∝ number of images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig_climate_separation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  fig_climate_separation.png")


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.project.seed)
    device = Device.auto(cfg.training.amp).torch()
    figures, reports = Path(cfg.paths.figures), Path(cfg.paths.reports)

    print("=" * 78)
    print("STAGE 7  |  Figures")
    print("=" * 78)
    fig_pathogen_response(figures)
    fig_network(figures, cfg)
    fig_climate_separation(cfg, figures)
    fig_backbones(reports, figures)
    fig_ablations(reports, figures)
    fig_horizon(reports, figures, cfg)
    fig_confusion(cfg, figures, device)
    print(f"\nfigures -> {figures}")


if __name__ == "__main__":
    main()
