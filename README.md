# Vision Transformer with Physics-Informed Graph Learning for Climate-Adaptive Crop Disease Forecasting

A complete, runnable implementation of the architecture in the project deck.
Every block in the architecture diagram maps to a module you can run and
demonstrate on its own.

**Animesh Gupta** · Panel Number: 0696 · Seva Innovation Challenge

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

`python scripts/06_serve.py` (or `09_go_live.py`, below) serves the dashboard at
<http://127.0.0.1:8000>, in five sections:

1. **Overview** - the problem, the approach, the dataset at a glance
2. **District Alerts** - the bulletin, the risk heatmap and the location view
3. **Live Diagnosis** - upload or load a leaf and run the full pipeline
4. **Project Status** - completion computed from artefacts on disk
5. **How it works** - press *Play* and one held-out test leaf goes through the
   live pipeline while an animated diagram lights up block by block (inputs ->
   vision transformer, disease weather, farm network -> fusion, graph network ->
   diagnosis, 7-day forecast -> what to do), each showing what it actually
   produced, with a short technical label per block (model, size, parameters)
   and the engine it ran on. The model runs in about 20 ms; the steps are paced
   for people to follow, and every value and timing shown is the server's

The literature survey, ablations and parameter tables remain in the API
(`/api/literature`, `/api/demo/*`, `/api/parameters`) and in `outputs/reports/`.

## For decision-makers: the district alert bulletin

The first live section of the dashboard is written for an agriculture officer,
not a researcher. Pick a crop and a date and every district growing it is
ranked by what needs attention this week:

- **Alert level** - *Act now*, *Watch* or *Normal*, from the worst day in the
  coming week, so a district that is quiet today but turns high in five days is
  flagged now
- **The threat** - which disease the weather favours there, and what kind of
  organism it is (fungal, water mould, bacterial, insect-borne, mite)
- **Direction** - whether risk has been rising or falling over the past week,
  with a sparkline of the past week and the coming one
- **Spread watch** - neighbouring districts on the spatial graph that are
  already on alert
- **Recommended actions** - for officials (SMS advisory, KVK field
  verification, input stock check) and for farmers, by organism type

Everything is computed from the weather actually recorded at each district,
through the same pathogen physics the network is trained against
(`serve/bulletin.py`, `GET /api/bulletin`). The 7-day outlook is a hindcast: it
replays the weather that followed, standing in for a weather forecast. Actions
are general guidance and defer product and dose to the state agriculture
department. A single diagnosed leaf gets the same treatment - the disease in
plain words, its spread risk, and what the farmer and the officials should do.

**Print bulletin** produces a paper copy of the bulletin alone.

### Risk heatmap

Beneath the district circles the map shades a 1 degree grid (about 110 km)
across everywhere within 350 km of a monitored district - 341 cells. Every cell
is computed exactly as a clicked location is: the weather recorded at that cell
through the pathogen physics, plus disease arriving on the wind
(`serve/heatmap.py`, `GET /api/heatmap`). Three layers - disease-favourable
weather, disease arriving, combined risk - each for today or the worst day of the
coming week. The surface is pinned to each district's own reading by
inverse-distance weighting, so at every district it equals that district's alert
exactly, and it is clipped to land. Hovering reads the value at any point;
clicking assesses that spot.

Grid weather is one archive lookup per cell per date, fetched in batches and
cached in `data/climate/grid/`; `09_go_live.py` prefetches the demo dates. The
free archive allows about 600 lookups a minute, so stepping through several new
dates quickly can briefly hit the limit - the page says so and retries.

### Why this result: leaf, weather, farm

Every leaf diagnosis carries the chain that produced it (`serve/context.py`):
what the diagnosed disease needs (its cardinal temperatures and moisture
requirement from the pathogen knowledge base), the weather actually recorded at
the farm that day, how favourable that makes it over the coming week, whether
the disease is present at the farm or arriving from neighbours, and the
resulting 7-day forecast. A note reconciles it with the district map, which can
say *Act now* (this week's weather favours the crop's diseases) while the farm's
forecast is lower (the forecast target needs favourable weather **and** the
disease nearby - `data.assign.attach_forecast_targets`). After a run the map
moves to that farm and date and opens its card.

The forecast's Low / Medium / High is read off the risk number with the
cut-offs the level head was trained on (`data/processed/risk_thresholds.json`),
so a label can never contradict its percentage.

The demo leaves in `tests/` are each paired with a farm and an in-season date
where the weather recorded there favoured that leaf's own disease - the way the
benchmark assigned leaves to farms - so consecutive runs are different places
and different weather (`scripts/pair_test_images.py`).

### Near a precise location

Switch the bulletin from **Whole country** to **Near a location** and choose a
point by GPS, by searching a village or town, or by clicking the map. The page
then answers the two questions behind the project's forecasting target
(`physics.contagion.combined_risk`) for that exact spot:

- **Is the weather here favourable?** The weather recorded at those
  coordinates, fetched from the same Open-Meteo archive as every district
  (no key), through the same agro-meteorology and pathogen physics
- **Is disease arriving?** The simulated epidemic's infection pressure at every
  district growing the crop, carried here by the contagion model's dispersal
  kernel - decaying with distance (180 km), boosted when the wind at the source
  was blowing this way

The location gets the same alert, trend, threat and actions as a district,
plus a ranked list of where disease could come from - distance, direction,
outbreak state, whether the wind is carrying it this way, and its share of what
is arriving (`serve/nearby.py`, `GET /api/nearby`, `GET /api/geocode`). If the
weather lookup is unavailable, the nearest district's weather is used and the
page says so.

**Try an example** loads a real case from the record: on 22 June 2023 Bhopal is
not a monitored district and its own weather is only moderately favourable
(22% today), but an outbreak is starting in Ujjain and Indore 170 km to the
west with the wind blowing east - so it is already flagged *Act now*. Three
days later its risk is 94%.

## Going live: frontend on Netlify, model on your machine

The model stays on the presenter's PC (it needs the GPU and the trained
weights); only the static page is hosted. One command brings the backend up and
makes it reachable from anywhere:

```bash
python scripts/09_go_live.py
```

It starts the API, warms the model up with one real prediction so the first run
in front of an audience is as fast as every later one, opens a Cloudflare quick
tunnel (`cloudflared`, no account needed), and prints the link to share:

```
SHARE THIS LINK  : https://<site>.netlify.app/?api=https://<random>.trycloudflare.com
```

The tunnel URL changes on every launch, so it travels in the link as `?api=`
rather than being baked into the Netlify build. The page remembers the last
backend it connected to, and if it cannot reach one it shows a box to paste a
backend URL into. The tunnel URL on its own also serves the full dashboard,
straight from the presenter's machine, as a fallback.

**Deploying the frontend on Netlify (once).** `netlify.toml` publishes
`src/cropforecast/serve/web` with no build step, and `web/_redirects` maps the
`/static/` prefix FastAPI uses, so the same `index.html` works on both hosts.

1. Sign in at <https://app.netlify.com> (a free account is enough).
2. **Add new site -> Import an existing project -> GitHub**, and pick this
   repository. Branch `main`. Leave the build command empty; the publish
   directory is read from `netlify.toml`. Press **Deploy**.
3. **Site configuration -> Change site name** to something short, e.g.
   `crop-early-warning-0696`, giving `https://crop-early-warning-0696.netlify.app`.
4. Put that URL in `configs/default.yaml` under `serve.frontend_url`, so
   `09_go_live.py` prints the complete share link.

Every push to `main` redeploys the page automatically.

**Or drag and drop.** `frontend_deploy/` at the project root is the same static
site, ready to drag onto <https://app.netlify.com/drop> (sign in first, or the
site expires after an hour). It is rebuilt from `src/cropforecast/serve/web` by
`python scripts/11_build_frontend.py`, and `09_go_live.py` rebuilds it on every
launch, so it always matches the page being served. Re-drag it only when the
page itself changes - a new tunnel needs no new upload, because the backend
travels in the link as `?api=`.

**On demo day.** Run `python scripts/09_go_live.py`, wait for `SHARE THIS LINK`,
and open or share that link - the Netlify page with `?api=<tunnel>` appended.
The tunnel address changes every launch, so share the link printed *that* time.
If the page says "backend offline", paste the printed tunnel URL into the box
in its header.

### Checking the demo before presenting

    python scripts/10_audit_demo.py                      # local server
    python scripts/10_audit_demo.py --url <tunnel URL>   # what the panel will hit

Runs every demo leaf through the single-shot and streamed pipelines and
cross-checks the answer against everything else describing that farm and day -
the bulletin card, the heatmap, the location view, the "why this result" chain -
and against `tests/index.csv`: about 800 checks, exit code 1 on any mismatch.

Two cautions the dashboard raises on its own, both measured with this audit's
data: an **off-season date** (across the 23 demo leaves at their farms through
2023, 99.4% correct in season vs 88.5% off season - off season the model drifts
toward whatever crop *is* in season there), and a **farm that does not grow the
crop**. Stating the crop in Live Diagnosis restricts the diagnosis to that
crop's diseases, which recovers the off-season cases.

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
scripts/                     01..08 run in order; 09_go_live.py for a public demo
netlify.toml                 static frontend hosting
outputs/                     figures, checkpoints, reports
```

---

## Data sources

- **PlantVillage** — Hughes & Salathé; `mohanty/PlantVillage` on HuggingFace, CC BY-SA 3.0
- **Open-Meteo Historical Archive** — ERA5 reanalysis, free, no API key
