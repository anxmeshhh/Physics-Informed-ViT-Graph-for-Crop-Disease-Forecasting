# Vision Transformer with Physics-Informed Graph Learning for Climate-Adaptive Crop Disease Forecasting

A complete, runnable implementation of the architecture in the project deck.
Every block in the architecture diagram maps to a module you can run and
demonstrate on its own.

**Team** — Animesh Gupta (RA2311026010375), Apurva Singh (RA2311026010376),
Aastha Hotwani (RA2311026010389), Harsh Khetan (RA2311026010421)

---

## What this is, precisely

Existing crop-disease systems classify a leaf photograph *after* symptoms appear.
This pipeline fuses three sources of evidence — the leaf image, the real
meteorology at that farm, and the disease state of neighbouring farms — and
forecasts disease risk **1, 3, 5 and 7 days ahead**.

### An honest statement about the data

This is a **constructed benchmark**, and it is important to describe it accurately
when you present it.

| Component | Status |
|---|---|
| Leaf imagery — 54,305 PlantVillage photos, 38 classes, 14 crops | **Real** |
| Meteorology — 41,662 site-days of ERA5 reanalysis via Open-Meteo | **Real** |
| Farm locations — 37 Indian production districts, real coordinates | **Real** |
| The *pairing* of an image to a (farm, date) | **Constructed** |

PlantVillage ships no coordinates and no timestamps, so the geography and dates
had to come from somewhere. Each leaf is placed at a farm that genuinely grows
that crop, on a date inside that crop's real growing season, weighted by two
things: whether the **weather actually recorded there** suited the labelled
disease, and whether the pathogen had actually *reached* that farm in a simulated
epidemic (below). The draw is stochastic with a probability floor, so the mapping
is noisy and cannot be inverted.

This is what gives the climate branch real signal, and we measured that it does
not make the task trivial:

- climate features alone reach **43.9%** on the 38-class problem (chance 2.6%, majority 10.4%)
- so climate matters a great deal, but the image is still doing most of the work

### The epidemic layer, and why it had to be added

The first version of this benchmark placed every leaf independently, weighted
only by local weather. It produced a dataset in which **all** spatial correlation
flowed through weather — and since every node already observes its own weather
directly, a neighbour carried no information the node did not already have.

Measured end to end, the graph was worth **−0.02 macro-F1**, and it stayed
negative even when 90% of farms were denied a photograph:

| Farms with no photo | GraphSAGE F1 | No-graph MLP F1 | Graph gain |
|---|---|---|---|
| 0% | 0.9376 | 0.9559 | −0.018 |
| 50% | 0.7048 | 0.7279 | −0.023 |
| 90% | 0.5021 | 0.5229 | −0.021 |

The graph was not failing; there was nothing spatial to find. That is also not
what plant epidemics do, and not what slide 5 of the deck claims — *wind can
carry spores from one field to another*.

`physics/contagion.py` supplies the missing mechanism: a daily compartmental
epidemic per crop, run on the real farm network under the real recorded weather,
with an exponential dispersal kernel amplified when the wind at the source farm
genuinely blew toward the target. Constants are calibrated so that **arrivals
from other farms dominate local build-up** (γ = 2.0 versus β = 0.1), the regime
of wind-dispersed pathogens such as the rusts.

The forecast target is then the risk a farmer actually faces — *conduciveness*
(will the weather here favour infection?) **and** *inoculum* (has the pathogen
arrived?). The first half is local; the second can only be known by looking at
the neighbours. Measured on the simulated epidemic, at 3- and 7-day lags a
farm's neighbours predict its future state **better than its own current state
does** (r = 0.845 vs 0.834 at 3 days; 0.564 vs 0.541 at 7 days).

This is labelled as a simulation. The imagery, the meteorology and the farm
geography are all real; the epidemic dynamics linking them are modelled, using a
standard formulation with physically interpretable constants.

### The leakage finding

PlantVillage photographs each physical leaf **2.65 times on average** (up to 9.25
for soybean). A random split therefore scatters near-duplicate views of the same
leaf across train and test:

| Split | Leaf groups spanning splits | Images affected |
|---|---|---|
| Random | 5,881 | **35,246 (65%)** |
| Official PlantVillage | 134 | — |
| **Leaf-disjoint (ours)** | **0** | **0** |

Every headline number in this repo uses the leaf-disjoint split. The ablation
table reports the random-split number too, so you can see exactly how much
accuracy leakage manufactures.

---

## Setup

```bash
# CUDA build - all three must agree in version
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Two traps worth knowing about, both already handled in this repo:

- **`captum>=0.8` requires `torch>=2.3`** and will silently pull a CPU-only torch
  over your CUDA build. It is pinned to `0.7.0`.
- `transformers` probes for TensorFlow and Flax at import. If a JAX build that
  wants numpy≥2 is installed, that probe crashes. `src/cropforecast/__init__.py`
  sets `USE_JAX=0` before transformers is ever imported.

---

## Running the pipeline

```bash
python scripts/01_build_dataset.py        # images + real climate + epidemic -> observations
python scripts/02_extract_features.py     # frozen embeddings, all 4 backbones (~8 min)
python scripts/03_benchmark_backbones.py  # DINOv2 vs ViT vs Swin vs DeiT
python scripts/04_train_and_ablate.py     # final model + ablation study
python scripts/05_explain.py              # Grad-CAM, IG, SHAP, graph attribution
python scripts/08_missing_data.py         # graph value as observations get sparse
python scripts/07_figures.py              # figures for the report
python scripts/06_serve.py                # web report at http://127.0.0.1:8000
```

Stage 1 downloads ~37 Open-Meteo requests once and caches them to
`data/climate/`; every later run is fully offline.

---

## Architecture, block by block

Each row is independently demonstrable — useful for splitting the work.

| # | Block | Module | Demo |
|---|---|---|---|
| 1 | Input layer | `data/farms.py`, `data/climate.py`, `data/assign.py` | `notebooks/01_dataset.ipynb` |
| 2a | Vision backbones | `models/backbones.py` | `notebooks/02_backbones.ipynb` |
| 2b | Climate features | `features/agromet.py` | `notebooks/03_climate.ipynb` |
| 2c | Graph construction | `graph/build.py` | `notebooks/04_graph.ipynb` |
| 3 | Feature fusion | `models/fusion.py` | `notebooks/05_model.ipynb` |
| 4 | GNN module | `models/gnn.py` | `notebooks/05_model.ipynb` |
| 5 | Prediction + physics loss | `models/heads.py`, `physics/losses.py` | `notebooks/06_physics.ipynb` |
| 6 | Explainability | `explain/` | `notebooks/07_explain.ipynb` |
| — | Epidemic spread | `physics/contagion.py` | `/api/demo/contagion` |

### The agronomic knowledge base

`physics/epidemiology.py` encodes the published environmental response of every
pathogen in the dataset, using the Analytis beta model for temperature and a
moisture term whose *form depends on the organism's biology*. Several responses
run in opposite directions, which is what makes the climate branch informative:

| Class | Optimum | Moisture mode |
|---|---|---|
| Potato / Tomato late blight | 17 °C | needs leaf wetness |
| Apple scab | 18 °C | needs leaf wetness (Mills periods) |
| Squash powdery mildew | 27 °C | humid air, **free water suppresses it** |
| Tomato spider mites | 31 °C | **hot and dry** |
| Tomato yellow leaf curl virus | 30 °C | hot and dry (whitefly vector) |
| Bacterial spots | 26–28 °C | rain splash + wind |

Verified on real Shimla weather: on the wettest day (17 °C, 90% RH, 20 h leaf
wetness) late blight scores 1.000 and powdery mildew 0.088; on the driest day
(22 °C, 28% RH) every fungal score collapses to 0.000 while spider mites rise
to 0.433.

### Avoiding circularity

The knowledge base is used **quantitatively** to build the benchmark, but the
physics loss never sees it. The loss only enforces weak directional statements:

- risk must not *fall* as leaf wetness and humidity rise (one-sided penalty)
- risk must stay bounded under strongly drying air (VPD > 1.5 kPa)
- risk must vary smoothly across connected farms (graph Dirichlet energy)
- forecast uncertainty must not *shrink* with a longer horizon

The network therefore has to learn the quantitative relationship from data.

---

## The web report

`python scripts/06_serve.py` serves a document-style report at
<http://127.0.0.1:8000>. It opens with the **18-paper literature survey**, and
every row carries a *Run demonstration* button that executes the corresponding
capability against the live system — the graph being built, the physics
constraints being evaluated, the epidemic spreading, the forecast running. The
survey stops being a table and becomes an argument you can audit.

Later sections cover the consolidated research gap, the methodology block by
block, the dataset and its honest provenance, single-image inference, the
regional risk map, and the results tables.

## The main negative result, stated plainly

**The graph neural network does not improve accuracy on this benchmark.**
Replacing the graph encoder with a plain MLP over the same fused features raises
both macro-F1 and forecast R². This held across two independently constructed
versions of the dataset, at five levels of observation sparsity, and for both
GraphSAGE and GCN.

| Configuration | Macro-F1 | Risk R² |
|---|---|---|
| Full (ViT + climate + graph + physics) | 0.9463 | 0.695 |
| **No graph (MLP encoder)** | **0.9835** | **0.782** |
| No climate stream | 0.9440 | 0.643 |
| No vision stream | 0.5107 | 0.738 |

Two things explain it, and both are properties of the problem rather than bugs:

1. **No headroom on diagnosis.** A frozen ViT-B/16 reaches ~98.8% accuracy on the
   leaf-disjoint test set from the image alone. Averaging in neighbouring farms —
   which carry *different* diseases — can only dilute an almost-saturated signal.
2. **A farm's own state is nearly a sufficient statistic.** In the simulated
   epidemic a farm's current infection pressure predicts its pressure three days
   later at r = 0.83; its neighbours reach r = 0.85. Better, but the *incremental*
   information beyond the farm's own state is small — and the node already
   observes that directly through its leaf image and its own weather.

The graph is kept as the default configuration because it is the architecture
this project set out to implement, and because its cost is now measured rather
than assumed. What the ablation table gives you is the honest price of the
design, which is a more defensible thing to present than an unverified claim.

The other streams behave exactly as the architecture predicts: **vision dominates
diagnosis** (removing it collapses F1 from 0.98 to 0.51) and **climate dominates
forecasting** (removing it costs 0.14 R²).

## Results

Generated into `outputs/reports/`:

- `stage3_backbone_benchmark.csv` — the four transformers, linear probe and full pipeline
- `stage4_ablations.csv` — no-graph, no-physics, no-climate, no-vision, GCN/GAT, and the leaky-split comparison
- `stage8_missing_data.csv` — what the graph is worth as observations become sparse

---

## Layout

```
configs/default.yaml         every tunable parameter
data/raw/                    PlantVillage (54,305 images)
data/climate/                cached Open-Meteo archive, one CSV per site
data/processed/              observations table + cached embeddings
src/cropforecast/
  config.py                  config loading, seeding, device
  data/                      farms, climate, plantvillage, assign, splits, dataset
  features/                  agromet (derived weather), tabular (scaling)
  graph/build.py             KNN + radius + wind-aware spatio-temporal graph
  models/                    backbones, fusion, gnn, heads, full_model
  physics/                   epidemiology (knowledge base), losses (constraints)
  explain/                   vision (Grad-CAM, rollout), attribution (IG, SHAP, graph)
  train/                     prepare, trainer
  serve/                     FastAPI api.py + inference.py + web/ dashboard
scripts/                     01..06, run in order
outputs/                     figures, checkpoints, reports
```

---

## Data sources

- **PlantVillage** — Hughes & Salathé; `mohanty/PlantVillage` on HuggingFace, CC BY-SA 3.0
- **Open-Meteo Historical Archive** — ERA5 reanalysis, free, no API key
