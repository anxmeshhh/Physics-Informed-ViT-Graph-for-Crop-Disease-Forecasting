"""Generate the per-stage demonstration notebooks.

One notebook per architecture block, so the pipeline can be walked through
"one by one" in a viva or a presentation. Regenerate at any time with:

    python scripts/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

BOOT = """\
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent / "src"))

import cropforecast
from cropforecast.config import load_config, ensure_dirs, set_seed, Device
cfg = load_config(Path.cwd().parent / "configs" / "default.yaml")
ensure_dirs(cfg); set_seed(cfg.project.seed)
device = Device.auto(cfg.training.amp)
print("device:", device)"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS: dict[str, list[dict]] = {}

# ---------------------------------------------------------------- 01 dataset
NOTEBOOKS["01_dataset.ipynb"] = [
    md("""# 01 - Input Layer

**Architecture block 1.** Leaf image + climate data + farm location & metadata.

PlantVillage has no coordinates and no dates. This notebook shows how the
geography and meteorology are attached, and verifies that the result is a
genuine learning problem rather than a giveaway."""),
    code(BOOT),

    md("## The farm registry\n\n37 real Indian production districts. These are the nodes of the spatial graph."),
    code("""from cropforecast.data.farms import SITES, to_frame, all_crops, sites_for_crop
sites = to_frame()
print(f"{len(sites)} sites across {sites.state.nunique()} states")
print(f"latitude {sites.lat.min():.2f} to {sites.lat.max():.2f}")
sites.head(10)"""),

    code("""for c in all_crops():
    print(f"{c:28s} {len(sites_for_crop(c)):2d} sites")"""),

    md("## The PlantVillage index\n\nNote `leaf_group`: PlantVillage photographs the same physical leaf several times."),
    code("""from cropforecast.data.plantvillage import build_index, class_table
index = build_index(cfg.paths.raw_images, cfg.paths.splits, cfg.paths.leaf_map)
print(f"{len(index):,} images | {index.class_name.nunique()} classes | "
      f"{index.leaf_group.nunique():,} leaf groups")
print(f"average {len(index)/index.leaf_group.nunique():.2f} photographs per physical leaf")
class_table(index).head(12)"""),

    md("""### Why leaf grouping matters

A random split scatters near-duplicate photographs of the *same leaf* across
train and test. Measure how bad it is."""),
    code("""from cropforecast.data.splits import leaf_disjoint_split, random_split, leakage_report
import pandas as pd
obs = pd.read_parquet(Path(cfg.paths.processed) / "observations.parquet")
print("random split      :", leakage_report(random_split(obs, seed=cfg.project.seed)))
print("leaf-disjoint     :", leakage_report(leaf_disjoint_split(obs, seed=cfg.project.seed)))"""),

    md("## Real climate\n\nERA5 reanalysis pulled once from the Open-Meteo archive and cached."),
    code("""from cropforecast.data.climate import fetch_all, coverage_report
climate_raw = fetch_all("2020-12-01", "2023-12-31", cfg.paths.climate,
                        daily_vars=list(cfg.climate.daily_vars), verbose=False)
print(f"{len(climate_raw):,} site-days")
coverage_report(climate_raw).head(12)"""),

    md("## Does the assignment create real signal?\n\nMean weather per disease class. Note the opposite responses."),
    code("""from cropforecast.data.assign import assignment_report
rep = assignment_report(obs)
pd.concat([rep.head(6), rep.tail(6)])[["n","temp_C","rh_pct","rain_mm","lwd_h","vpd"]]"""),

    md("""**Read the table.** Late blight and strawberry leaf scorch sit at ~16 C with
10-12 h of leaf wetness. Spider mites and corn sit at ~28 C with high VPD. That
contrast is what the climate branch learns from."""),
]

# -------------------------------------------------------------- 02 backbones
NOTEBOOKS["02_backbones.ipynb"] = [
    md("""# 02 - Vision Transformers

**Architecture block 2a.** Four transformer families behind one interface:
DINOv2-S (self-supervised, the primary backbone), ViT-B/16 (supervised),
Swin-T (hierarchical) and DeiT-S (distilled)."""),
    code(BOOT),

    code("""import torch
from cropforecast.models.backbones import SPECS, load_backbone
for k, s in SPECS.items():
    print(f"{k:8s} {s.hf_id:42s} dim={s.dim:4d} pool={s.pool:5s} {s.family}")"""),

    md("""### Patch embedding and self-attention

The diagram's steps 3-4: split the image into patches, embed them, add
positional encoding, then run L encoder blocks of multi-head self-attention."""),
    code("""bb = load_backbone("dinov2", frozen=True).to(device.name).eval()
x = torch.randn(2, 3, 224, 224, device=device.name)
with torch.no_grad():
    out = bb(x)
print("pooled (CLS) :", tuple(out["pooled"].shape))
print("patch tokens :", tuple(out["tokens"].shape), "-> 1 CLS + 16x16 patches")
print("frozen params:", sum(p.numel() for p in bb.parameters())/1e6, "M")"""),

    md("### A real leaf through the backbone"),
    code("""import numpy as np, pandas as pd
from PIL import Image
from transformers import AutoImageProcessor
obs = pd.read_parquet(Path(cfg.paths.processed) / "observations.parquet")
row = obs[obs.class_name == "Tomato___Late_blight"].iloc[0]
img = Image.open(row.image_path).convert("RGB").resize((224, 224))

proc = AutoImageProcessor.from_pretrained(SPECS["dinov2"].hf_id)
mean = torch.tensor(proc.image_mean, device=device.name).view(1,3,1,1)
std  = torch.tensor(proc.image_std,  device=device.name).view(1,3,1,1)
t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2,0,1)[None].to(device.name).float()/255
t = (t - mean)/std
with torch.no_grad():
    emb = bb(t)["pooled"]
print(row.class_name, "->", tuple(emb.shape))
display(img)"""),

    md("### Attention rollout: where is the transformer looking?"),
    code("""from cropforecast.explain.vision import attention_rollout, overlay
import matplotlib.pyplot as plt
bb_attn = load_backbone("dinov2", frozen=True, output_attentions=True).to(device.name).eval()
roll = attention_rollout(bb_attn, t)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(np.array(img)); ax[0].set_title("input"); ax[0].axis("off")
ax[1].imshow(overlay(np.array(img), roll)); ax[1].set_title("attention rollout"); ax[1].axis("off")
plt.show()"""),

    md("### Benchmark results\n\nRun `python scripts/03_benchmark_backbones.py` first."),
    code("""bench = Path(cfg.paths.reports) / "stage3_backbone_benchmark.csv"
pd.read_csv(bench) if bench.exists() else print("Run scripts/03_benchmark_backbones.py")"""),
]

# ---------------------------------------------------------------- 03 climate
NOTEBOOKS["03_climate.ipynb"] = [
    md("""# 03 - Climate Feature Engineering

**Architecture block 2b.** Raw climate -> scaling -> temporal aggregation ->
agricultural feature set.

The derived variables are the ones plant pathologists actually use, because
raw temperature alone does not drive infection."""),
    code(BOOT),

    code("""import pandas as pd, numpy as np
from cropforecast.features.agromet import build_climate_features, derive_daily_features
raw = pd.read_csv(Path(cfg.paths.climate) / "HP-SML.csv", parse_dates=["date"])
feat = build_climate_features(raw, windows=tuple(cfg.climate.rolling_windows))
print(f"{raw.shape[1]} raw columns -> {feat.shape[1]} engineered columns")
[c for c in feat.columns if c not in raw.columns][:25]"""),

    md("""### The derived quantities

- **VPD** - drying power of the air (Tetens equation)
- **Leaf wetness hours** - the single best predictor of fungal infection
- **Dew-point depression** - how close the air is to condensing
- **GDD** - crop development clock
- **wind_u / wind_v** - spore transport vector"""),
    code("""import matplotlib.pyplot as plt
sub = feat[feat.date.dt.year == 2023]
fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
ax[0].plot(sub.date, sub.temperature_2m_mean, color="#f43f5e"); ax[0].set_ylabel("Temp C")
ax[1].plot(sub.date, sub.relative_humidity_2m_mean, color="#38bdf8"); ax[1].set_ylabel("RH %")
ax[2].plot(sub.date, sub.leaf_wetness_hours, color="#22c55e"); ax[2].set_ylabel("Leaf wet h")
ax[3].plot(sub.date, sub.vpd_kpa, color="#fbbf24"); ax[3].set_ylabel("VPD kPa")
fig.suptitle("Shimla 2023 - monsoon is unmistakable"); plt.tight_layout(); plt.show()"""),

    md("""## The agronomic knowledge base

Each pathogen has its own environmental response. Several are *opposites*."""),
    code("""from cropforecast.physics.epidemiology import PROFILES, temperature_response
import numpy as np
t = np.linspace(0, 42, 300)
fig, ax = plt.subplots(figsize=(11, 5))
for name in ["Potato___Late_blight","Apple___Apple_scab","Squash___Powdery_mildew",
             "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Bacterial_spot"]:
    ax.plot(t, temperature_response(t, PROFILES[name]), label=name.split("___")[1].replace("_"," "), lw=2)
ax.set_xlabel("Temperature C"); ax.set_ylabel("relative development rate")
ax.set_title("Analytis beta model - cardinal temperatures per pathogen")
ax.legend(); ax.grid(alpha=.3); plt.show()"""),

    md("### Favourability on real weather\n\nWettest vs driest recorded day at Shimla."),
    code("""from cropforecast.physics.epidemiology import favourability
wet = feat.loc[[feat.leaf_wetness_hours.idxmax()]]
dry = feat.loc[[feat.vpd_kpa.idxmax()]]
classes = ["Apple___Apple_scab","Potato___Late_blight","Squash___Powdery_mildew",
           "Tomato___Spider_mites Two-spotted_spider_mite","Apple___healthy"]
pd.DataFrame({
    "wettest day": [favourability(c, wet)[0] for c in classes],
    "driest day":  [favourability(c, dry)[0] for c in classes],
}, index=classes).round(3)"""),

    md("""Powdery mildew collapses on the wet day (free water bursts its spores) while
late blight peaks. Spider mites do the reverse. This is the structure that makes
the climate stream worth having."""),
]

# ------------------------------------------------------------------ 04 graph
NOTEBOOKS["04_graph.ipynb"] = [
    md("""# 04 - Graph Construction

**Architecture block 2c.** Combine location + time -> KNN + radius search ->
spatial graph.

A node is one observation. Edges join observations close in space *and* time,
plus directed downwind edges for wind-borne spore transport."""),
    code(BOOT),

    code("""from cropforecast.graph.build import haversine_km, initial_bearing_deg, site_distance_matrix
from cropforecast.data.farms import to_frame
sites = to_frame()
d, b, ids = site_distance_matrix(sites)
print(f"Nashik -> Sangli : {haversine_km(19.9975,73.7898,16.8524,74.5815):.1f} km (real ~360)")
print(f"bearing          : {initial_bearing_deg(19.9975,73.7898,16.8524,74.5815):.0f} deg")"""),

    md("### Calibrating the radius\n\nIndian production belts are far apart, so a small radius disconnects the graph."),
    code("""import numpy as np
rows = []
for r in [150, 250, 350, 500]:
    A = (d < r) & (d > 0)
    rows.append({"radius_km": r, "mean_degree": A.sum(1).mean().round(2),
                 "isolated_sites": int((A.sum(1) == 0).sum())})
import pandas as pd; pd.DataFrame(rows)"""),

    md("""At 150 km a third of the sites are orphaned. We use 350 km **plus** a KNN
fallback (`min_neighbours`) so no farm is ever isolated - the diagram does say
"KNN *and* radius search"."""),

    code("""from cropforecast.graph.build import build_graph
obs = pd.read_parquet(Path(cfg.paths.processed) / "observations.parquet")
sample = obs.sample(4000, random_state=cfg.project.seed).reset_index(drop=True)
g = build_graph(sample, sites,
                k_neighbours=cfg.graph.k_neighbours, radius_km=cfg.graph.radius_km,
                time_window_days=cfg.graph.time_window_days,
                max_edges_per_node=cfg.graph.max_edges_per_node,
                wind_aware=True, min_neighbours=cfg.graph.min_neighbours)
g.summary()"""),

    md("### Downwind edges\n\nDirected edges added where the wind at the source actually blew toward the target."),
    code("""print("edge attributes:", g.ATTR_NAMES)
print(f"downwind edges: {(g.edge_type==1).sum():,} of {g.edge_index.shape[1]:,}")
pd.DataFrame(g.edge_attr[:8], columns=list(g.ATTR_NAMES)).round(3)"""),


    md("""## Wind-borne spread between farms

The graph only has something to learn if a farm's future depends on its
*neighbours*, not just on itself. This is the epidemic that supplies that
dependence."""),
    code("""from cropforecast.physics.contagion import simulate_crop, spatial_autocorrelation, EpidemicParams
climate = pd.read_parquet(Path(cfg.paths.processed) / "climate_features.parquet")
climate["date"] = pd.to_datetime(climate["date"])
print("epidemic constants:", EpidemicParams())
sim = simulate_crop(climate, "Tomato")
print(f"{len(sim):,} crop-site-days simulated")
sim.infection_pressure.describe().round(4)"""),

    md("""### The decisive measurement

Compare two predictors of a farm's infection pressure *h* days ahead: its own
current pressure, and its neighbours' current pressure."""),
    code("""rows = [spatial_autocorrelation(sim, climate, "Tomato", lag_days=h) for h in (1, 3, 7)]
pd.DataFrame(rows)"""),

    md("""At longer lags the neighbours are the *better* predictor. That is the signal a
GNN can exploit and a per-node model cannot."""),

    code("""# An outbreak travelling across the network
wide = sim.pivot(index="date", columns="site_id", values="infection_pressure")
peak = np.unravel_index(np.nanargmax(wide.to_numpy()), wide.shape)
window = wide.iloc[max(0, peak[0]-25):peak[0]+10]
fig, ax = plt.subplots(figsize=(13, 5))
for sid in window.columns:
    ax.plot(window.index, window[sid], lw=1.6, label=sid)
ax.set_ylabel("infection pressure"); ax.set_title("An epidemic moving between farms")
ax.legend(ncol=4, fontsize=8); ax.grid(alpha=.3); plt.show()"""),

    md("### The farm network"),
    code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 10))
k, radius, minnb = cfg.graph.k_neighbours, cfg.graph.radius_km, cfg.graph.min_neighbours
for i, sid in enumerate(ids):
    order = [j for j in d[i].argsort() if j != i]
    within = [j for j in order if d[i, j] <= radius][:k] or []
    if len(within) < minnb: within = order[:minnb]
    for j in within:
        ax.plot([sites.lon[i], sites.lon[j]], [sites.lat[i], sites.lat[j]],
                color="#22c55e", alpha=.25, lw=.8, zorder=1)
ax.scatter(sites.lon, sites.lat, s=70, c="#f43f5e", zorder=2, edgecolor="white")
for _, r in sites.iterrows():
    ax.annotate(r["name"], (r.lon, r.lat), fontsize=7, xytext=(3,3), textcoords="offset points")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Spatial graph over 37 Indian farm districts"); ax.grid(alpha=.2); plt.show()"""),
]

# ------------------------------------------------------------------ 05 model
NOTEBOOKS["05_model.ipynb"] = [
    md("""# 05 - Fusion + Graph Neural Network

**Architecture blocks 3-4.** Vision (+) climate (+) metadata -> fusion -> GNN
-> spatially aware node embeddings."""),
    code(BOOT),

    md("""## Fusion

Two modes. `concat` is the diagram's own concatenation/projection. `gated` adds
FiLM modulation so climate can *scale* the vision stream - the same lesion means
something different in a wet spell than a dry one."""),
    code("""import torch
from cropforecast.models.fusion import FusionModule
f = FusionModule(vision_dim=384, climate_dim=51, meta_dim=30, hidden_dim=256, mode="gated")
v, c, m = torch.randn(8,384), torch.randn(8,51), torch.randn(8,30)
print("fused:", tuple(f(v,c,m).shape), "| params:", sum(p.numel() for p in f.parameters())/1e3, "K")"""),

    md("## Graph encoder\n\nAll three convolutions from the diagram, plus a no-graph ablation."),
    code("""from cropforecast.models.gnn import build_encoder
x = torch.randn(200, 256); ei = torch.randint(0, 200, (2, 900)); ea = torch.rand(900, 4)
for conv in ["sage", "gcn", "gat", "none"]:
    enc = build_encoder(conv, 256, hidden_dim=256, num_layers=3, heads=4)
    print(f"{conv:5s} -> {tuple(enc(x, ei, ea).shape)}   params {sum(p.numel() for p in enc.parameters())/1e3:7.1f} K")"""),

    md("## The full model"),
    code("""from cropforecast.models.full_model import CropDiseaseForecastNet
net = CropDiseaseForecastNet(vision_dim=384, climate_dim=51, meta_dim=30,
                             num_classes=38, horizons=(1,3,5,7), gnn_conv="sage")
out = net(torch.randn(200,384), torch.randn(200,51), torch.randn(200,30), ei, ea)
for k, val in out.items(): print(f"  {k:14s} {tuple(val.shape)}")
print("trainable:", round(net.num_trainable()/1e6, 3), "M  (backbone is frozen and cached)")"""),

    md("""### Why the forecaster is autoregressive

Horizon *h* is predicted from the state carried forward from *h-1* through a GRU
cell. Independent heads have no way to know day 7 follows day 5 and produce
forecasts that jump around non-physically."""),
    code("""r = out["risk"][:5].detach()
import pandas as pd
pd.DataFrame(r.numpy(), columns=["+1d","+3d","+5d","+7d"]).round(3)"""),

    md("### Ablation results\n\nRun `python scripts/04_train_and_ablate.py` first."),
    code("""abl = Path(cfg.paths.reports) / "stage4_ablations.csv"
pd.read_csv(abl) if abl.exists() else print("Run scripts/04_train_and_ablate.py")"""),
]

# ---------------------------------------------------------------- 06 physics
NOTEBOOKS["06_physics.ipynb"] = [
    md("""# 06 - Physics-Informed Loss

**Architecture block 5.** `Loss = CE + lambda * Physics Loss`.

The physics loss never sees the forward epidemiological model - otherwise the
study would be circular. It states only weak directional facts."""),
    code(BOOT),

    code("""import torch
from cropforecast.physics.losses import (
    PhysicsWeights, physics_loss, monotonic_horizon_loss,
    diffusion_loss, dry_suppression_loss, PHYSICS_COLUMNS)
print("physics inputs:", PHYSICS_COLUMNS)
print("weights       :", PhysicsWeights())"""),

    md("### Constraint 1 - forecast uncertainty must not shrink with horizon"),
    code("""inc = torch.arange(4).float().repeat(16,1)     # uncertainty grows: legal
dec = -inc                                     # uncertainty shrinks: illegal
print("increasing uncertainty ->", monotonic_horizon_loss(inc).item(), "(no penalty)")
print("decreasing uncertainty ->", monotonic_horizon_loss(dec).item(), "(penalised)")"""),

    md("### Constraint 2 - risk is spatially smooth across connected farms"),
    code("""ei = torch.randint(0, 64, (2, 300)); ew = torch.rand(300)
uniform = torch.ones(64, 4)
noisy   = torch.rand(64, 4)
print("uniform risk across farms ->", round(diffusion_loss(uniform, ei, ew).item(), 5))
print("random risk across farms  ->", round(diffusion_loss(noisy,  ei, ew).item(), 5))"""),

    md("### Constraint 3 - dry air suppresses fungal risk"),
    code("""risk_high = torch.full((32,4), 0.9)
vpd_dry   = torch.full((32,), 2.0)    # strongly drying
vpd_humid = torch.full((32,), 0.3)
print("high risk + dry air   ->", round(dry_suppression_loss(risk_high, vpd_dry).item(), 4))
print("high risk + humid air ->", round(dry_suppression_loss(risk_high, vpd_humid).item(), 4))"""),

    md("### All terms together"),
    code("""torch.manual_seed(0)
risk = torch.rand(64,4, requires_grad=True); logvar = torch.randn(64,4, requires_grad=True)
phys = torch.stack([torch.rand(64)*20, 40+torch.rand(64)*55, torch.rand(64)*2.5, 10+torch.rand(64)*25],1)
total, parts = physics_loss(risk, logvar, phys, ei, ew)
print("total:", round(total.item(),4)); parts"""),

    md("### Training curve\n\nHow each physics term evolves. Run stage 4 first."),
    code("""import pandas as pd, matplotlib.pyplot as plt
hist_path = Path(cfg.paths.reports) / "stage4_history.csv"
if hist_path.exists():
    h = pd.read_csv(hist_path)
    cols = [c for c in h.columns if c.startswith("phys_")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for c in cols: ax[0].plot(h.epoch, h[c], label=c.replace("phys_",""))
    ax[0].set_title("physics constraint violations"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    ax[1].plot(h.epoch, h.val_risk_r2_mean, label="val risk R2")
    ax[1].set_title("validation"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()
else:
    print("Run scripts/04_train_and_ablate.py")"""),
]

# ---------------------------------------------------------------- 07 explain
NOTEBOOKS["07_explain.ipynb"] = [
    md("""# 07 - Explainability

**Architecture block 6.** Grad-CAM, SHAP, integrated gradients, graph
explainability and attention maps.

Requires a trained model: run `scripts/04_train_and_ablate.py` first."""),
    code(BOOT),

    code("""import torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from cropforecast.models.full_model import CropDiseaseForecastNet
from cropforecast.models.backbones import load_backbone, SPECS
from transformers import AutoImageProcessor

blob = torch.load(Path(cfg.paths.checkpoints) / "final_model.pt", map_location=device.name)
m, c = blob["meta"], blob["config"]
model = CropDiseaseForecastNet(
    vision_dim=m["vision_dim"], climate_dim=m["climate_dim"], meta_dim=m["meta_dim"],
    num_classes=m["num_classes"], horizons=tuple(m["horizons"]),
    fusion_dim=c["fusion_dim"], gnn_conv=c["gnn_conv"], gnn_hidden=c["gnn_hidden"],
    gnn_layers=c["gnn_layers"], gnn_heads=c["gnn_heads"]).to(device.name)
model.load_state_dict(blob["state_dict"]); model.eval()
print("loaded model trained on backbone:", c["backbone"])"""),

    md("## Visual explanations"),
    code("""from cropforecast.explain.vision import attention_rollout, grad_cam, overlay
bb = load_backbone(c["backbone"], frozen=True, output_attentions=True).to(device.name).eval()
proc = AutoImageProcessor.from_pretrained(SPECS[c["backbone"]].hf_id)
mean = torch.tensor(proc.image_mean, device=device.name).view(1,3,1,1)
std  = torch.tensor(proc.image_std,  device=device.name).view(1,3,1,1)

obs = pd.read_parquet(Path(cfg.paths.processed) / "observations.parquet")
row = obs[obs.class_name == "Tomato___Late_blight"].iloc[0]
img = Image.open(row.image_path).convert("RGB").resize((224,224))
arr = np.array(img, dtype=np.uint8)
x = torch.from_numpy(arr).permute(2,0,1)[None].to(device.name).float()/255
x = (x-mean)/std

roll = attention_rollout(bb, x)
fig, ax = plt.subplots(1,2, figsize=(9,4.5))
ax[0].imshow(arr); ax[0].set_title(row.class_name); ax[0].axis("off")
ax[1].imshow(overlay(arr, roll)); ax[1].set_title("attention rollout"); ax[1].axis("off")
plt.show()"""),

    md("## Integrated gradients over climate"),
    code("""ig_path = Path(cfg.paths.reports) / "stage5_integrated_gradients.csv"
if ig_path.exists():
    ig = pd.read_csv(ig_path).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(ig.feature, ig.attribution,
            color=["#f43f5e" if v<0 else "#22c55e" for v in ig.attribution])
    ax.axvline(0, color="#888", lw=.8); ax.set_xlabel("attribution to +1d risk")
    plt.tight_layout(); plt.show()
else:
    print("Run scripts/05_explain.py")"""),

    md("## Graph attribution\n\nWhich neighbouring farms drive one field's forecast?"),
    code("""ga = Path(cfg.paths.reports) / "stage5_graph_attribution.csv"
pd.read_csv(ga) if ga.exists() else print("Run scripts/05_explain.py")"""),

    md("## SHAP global importance"),
    code("""sh = Path(cfg.paths.reports) / "stage5_shap.csv"
pd.read_csv(sh).head(15) if sh.exists() else print("Run scripts/05_explain.py")"""),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = NB_DIR / name
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"  wrote {path.relative_to(ROOT)}  ({len(cells)} cells)")
    print(f"\n{len(NOTEBOOKS)} notebooks generated")


if __name__ == "__main__":
    main()
