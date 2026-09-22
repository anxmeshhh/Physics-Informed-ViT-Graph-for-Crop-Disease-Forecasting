"""FastAPI backend for the demonstration dashboard.

Start with:
    python -m uvicorn cropforecast.serve.api:app --reload --port 8000
or simply:
    python scripts/06_serve.py

The trained model is loaded once at startup and reused for every request.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ..config import PROJECT_ROOT, load_config
from ..data.farms import to_frame
from . import demos, literature
from .inference import ForecastService

WEB_DIR = Path(__file__).parent / "web"
SAMPLES_DIR = PROJECT_ROOT / "tests"

app = FastAPI(
    title="Climate-Adaptive Crop Disease Forecasting",
    description="Vision Transformer with Physics-Informed Graph Learning",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_cfg = load_config()
_service: ForecastService | None = None


def service() -> ForecastService:
    """Lazily construct the service so import never fails without a checkpoint."""
    global _service
    if _service is None:
        try:
            _service = ForecastService(_cfg)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=("Model not trained yet. Run scripts/04_train_and_ablate.py "
                        f"first. ({exc})"),
            ) from exc
    return _service


# --- static frontend -------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"message": "Frontend not found", "docs": "/docs"})
    return FileResponse(str(page))


# --- metadata endpoints ----------------------------------------------------
@app.get("/api/health")
def health():
    ready = (Path(_cfg.paths.checkpoints) / "final_model.pt").exists()
    return {"status": "ok", "model_trained": ready}


@app.get("/api/sites")
def sites():
    """The farm registry - nodes of the spatial graph."""
    df = to_frame()
    return {"count": len(df), "sites": df.to_dict(orient="records")}


@app.get("/api/crops")
def crops():
    svc = service()
    return {"crops": [{"crop": c, "season": svc.season_for(c)} for c in svc.crops()]}


# --- curated sample inputs -------------------------------------------------
_samples: list[dict] | None = None


def sample_index() -> list[dict]:
    """The curated demo set described by ``tests/index.csv``, read once.

    Each row already pairs an image with a farm that grows that crop and a date
    inside its season, so a one-click demonstration cannot produce a nonsense
    forecast. Rows whose image is missing are dropped rather than served as 404s.
    """
    global _samples
    if _samples is not None:
        return _samples

    index = SAMPLES_DIR / "index.csv"
    if not index.exists():
        _samples = []
        return _samples

    df = pd.read_csv(index)
    _samples = [
        {
            "file": str(r.file),
            "true_class": str(r.true_class),
            "crop": str(r.crop),
            "confidence": (None if pd.isna(r.confidence) else round(float(r.confidence), 4)),
            "site_id": str(r.suggested_site_id),
            "farm": str(r.suggested_farm),
            "date": str(r.suggested_date),
        }
        for r in df.itertuples()
        if (SAMPLES_DIR / str(r.file)).is_file()
    ]
    return _samples


@app.get("/api/samples")
def samples():
    """Demo images from the held-out test split, with a farm and date for each."""
    return {"count": len(sample_index()), "samples": sample_index()}


@app.get("/api/samples/{name}")
def sample_image(name: str):
    """Serve one curated image. Only names listed in the index are reachable."""
    if name not in {s["file"] for s in sample_index()}:
        raise HTTPException(status_code=404, detail=f"No sample image named {name!r}")
    return FileResponse(str(SAMPLES_DIR / name), media_type="image/jpeg")


@app.get("/api/summary")
def summary():
    """Headline numbers for the dashboard, read from the saved reports."""
    reports = Path(_cfg.paths.reports)
    out: dict = {"model_trained": (Path(_cfg.paths.checkpoints) / "final_model.pt").exists()}

    processed = Path(_cfg.paths.processed)
    if (processed / "observations.parquet").exists():
        obs = pd.read_parquet(processed / "observations.parquet",
                              columns=["class_name", "site_id", "date", "crop"])
        out["observations"] = len(obs)
        out["classes"] = int(obs.class_name.nunique())
        out["crops"] = int(obs.crop.nunique())
        out["sites"] = int(obs.site_id.nunique())
        out["date_range"] = [str(obs.date.min().date()), str(obs.date.max().date())]

    for name, key in [("stage3_backbone_benchmark.csv", "backbones"),
                      ("stage4_ablations.csv", "ablations")]:
        path = reports / name
        if path.exists():
            out[key] = pd.read_csv(path).round(4).to_dict(orient="records")
    return out


@app.get("/api/parameters")
def parameters():
    """Every parameter the system considers, assembled from the live config.

    Written out by hand this would drift away from the run the moment anyone
    edited the YAML, so each row is read from ``configs/default.yaml`` and from
    the fitted feature space instead. What the page shows is what the pipeline
    actually used.
    """
    b, g, m = _cfg.backbones, _cfg.graph, _cfg.model
    p, t = _cfg.physics, _cfg.training
    primary = b[b.primary]
    frozen = "frozen" if b.frozen else "fine-tuned"

    # Stream widths are only known once stage 4 has fitted the feature space.
    climate_dim: int | str = "pending stage 4"
    meta_dim: int | str = "pending stage 4"
    space_path = Path(_cfg.paths.checkpoints) / "feature_space.pkl"
    if space_path.exists():
        try:
            with open(space_path, "rb") as fh:
                space = pickle.load(fh)
            climate_dim, meta_dim = space.climate_dim, space.meta_dim
        except Exception:                                    # noqa: BLE001
            pass

    horizons = "/".join(str(h) for h in m.heads.horizons)
    windows = "/".join(str(w) for w in _cfg.climate.rolling_windows)

    groups = [
        {
            "id": "observed",
            "title": "What the model observes",
            "lead": "Three streams, per farm per day. Every one is ablated "
                    "separately, so its contribution is measured rather than assumed.",
            "rows": [
                ["Leaf image", f"1 RGB photo, {primary.image_size}px",
                 f"Encoded by {b.primary} ({frozen}) to a {primary.dim}-d embedding"],
                ["Raw weather variables", f"{len(_cfg.climate.daily_vars)} daily",
                 "ERA5 reanalysis via Open-Meteo, recorded at that farm on that date"],
                ["Derived agro-meteorology",
                 "VPD, leaf wetness, GDD, dewpoint depression, diurnal range, wind u/v",
                 "The quantities plant pathologists use, not raw weather"],
                ["Temporal aggregation", f"{windows}-day rolling windows",
                 f"Disease responds to accumulated conditions, not one day; "
                 f"{_cfg.climate.lookback_days}-day lookback"],
                ["Climate stream width", climate_dim,
                 "Raw + derived + rolling, after excluding targets and identifiers"],
                ["Farm metadata",
                 "lat, lon, elevation, crop, soil type, agro-zone, season sin/cos",
                 f"{meta_dim}-d after scaling and one-hot encoding"],
            ],
        },
        {
            "id": "neighbourhood",
            "title": "How a neighbourhood is defined",
            "lead": "Which other farms a node is allowed to see. These decide what "
                    "the graph can possibly learn.",
            "rows": [
                ["Neighbours per farm", f"k = {g.k_neighbours}",
                 f"KNN, capped at {g.max_edges_per_node} edges per node"],
                ["Search radius", f"{g.radius_km:g} km",
                 "Indian production belts are far apart; calibrated for connectivity"],
                ["Temporal window", f"+/- {g.time_window_days} days",
                 "Edges are spatio-temporal, not purely spatial"],
                ["Wind-aware edges", "on" if g.wind_aware else "off",
                 f"Directed downwind edge when cos(angle) > "
                 f"{g.wind_alignment_threshold}, for spore transport"],
                ["Crop isolation", "on" if g.same_crop else "off",
                 "A pathogen does not cross between different crops"],
                ["Isolation floor", f"min {g.min_neighbours} neighbours",
                 "KNN fallback so no farm is ever left without context"],
            ],
        },
        {
            "id": "architecture",
            "title": "Architecture and capacity",
            "lead": "Everything downstream of the frozen backbone is trained, and "
                    "kept deliberately small - the features are already strong.",
            "rows": [
                ["Fusion width", f"{m.fusion.hidden_dim}-d",
                 f"Common width for all three streams; dropout {m.fusion.dropout}"],
                ["Graph encoder", m.gnn.conv,
                 "sage | gcn | gat - all three implemented, all reported"],
                ["Message-passing depth", f"{m.gnn.num_layers} layers",
                 "2 hops; 3 over-smooths"],
                ["GNN width", f"{m.gnn.hidden_dim}-d",
                 f"Dropout {m.gnn.dropout}; {m.gnn.heads} attention heads if GAT"],
                ["Classification head", f"{m.heads.num_classes} classes",
                 "The diagnosis for the leaf in front of you"],
                ["Forecast horizons", f"{horizons} days",
                 f"Multi-horizon head, {m.heads.risk_levels} risk bands"],
            ],
        },
        {
            "id": "physics",
            "title": "Physics constraints",
            "lead": f"Total weight lambda = {p.lambda_physics}. The loss enforces "
                    "direction only - never the quantitative knowledge base - so the "
                    "network still has to learn the relationship from data.",
            "rows": [
                ["Infection pressure", f"{p.weights.infection_pressure:g}",
                 "Risk must not fall as leaf wetness and humidity rise"],
                ["Dry suppression", f"{p.weights.dry_suppression:g}",
                 "Risk stays bounded under strongly drying air (VPD > 1.5 kPa)"],
                ["Diffusion", f"{p.weights.diffusion:g}",
                 "Risk varies smoothly across connected farms (Dirichlet energy)"],
                ["Monotonic horizon", f"{p.weights.monotonic_horizon:g}",
                 "Forecast uncertainty must not shrink as the horizon lengthens"],
            ],
        },
        {
            "id": "training",
            "title": "Training protocol",
            "lead": "Full-batch over the graph. The split policy matters more here "
                    "than any hyperparameter.",
            "rows": [
                ["Split", "leaf-disjoint",
                 "0 leaf groups span train and test; a random split leaks 65% of images"],
                ["Epochs", t.epochs, "Full-batch graph training, ~0.2 s per epoch"],
                ["Learning rate", f"{t.lr:g}", f"Weight decay {t.weight_decay:g}"],
                ["Batch size", t.batch_size,
                 "Feature extraction; the GNN itself is full-batch"],
                ["Mixed precision", "on" if t.amp else "off", "Fits in 6 GB of VRAM"],
                ["Seed", _cfg.project.seed,
                 "Assignment, splits and initialisation are all seeded"],
                ["Study period",
                 f"{_cfg.data.study_years[0]}-{_cfg.data.study_years[-1]}",
                 f"{_cfg.data.season_window_days}-day growing-season sampling window"],
            ],
        },
        {
            "id": "reported",
            "title": "What we report",
            "lead": "Chosen before the results were in. Macro-F1 rather than accuracy, "
                    "because the 38 classes are heavily imbalanced.",
            "rows": [
                ["Macro-F1", "38-class diagnosis",
                 "Unweighted mean over classes; the majority class is only 10.4%"],
                ["Accuracy", "38-class diagnosis", "Reported alongside, never alone"],
                ["Risk R-squared", f"per horizon ({horizons} d)",
                 "Variance in future risk that the model explains"],
                ["Risk MAE", f"per horizon ({horizons} d)",
                 "Absolute error on the 0-1 risk"],
                ["Band accuracy", f"{m.heads.risk_levels} bands",
                 "What a farmer actually acts on: low / medium / high"],
            ],
        },
    ]
    return {"config_file": "configs/default.yaml", "groups": groups}


@app.get("/api/climate/{site_id}")
def climate(site_id: str, start: str = "2023-01-01", end: str = "2023-12-31"):
    svc = service()
    try:
        return svc.climate_series(site_id, Date.fromisoformat(start),
                                  Date.fromisoformat(end))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/forecast")
def regional(crop: str, date: str):
    """Risk map across every farm growing this crop on a date."""
    svc = service()
    try:
        when = Date.fromisoformat(date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from exc
    rows = svc.regional_forecast(when, crop)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No climate data for {crop} on {date}. Try a 2021-2023 date.",
        )
    return {"crop": crop, "date": date, "sites": rows}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    site_id: str = Form(...),
    date: str = Form(...),
    crop: str | None = Form(None),
):
    """Full pipeline on one uploaded leaf photograph."""
    svc = service()
    try:
        when = Date.fromisoformat(date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from exc

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {exc}") from exc

    try:
        pred = svc.predict(img, site_id, when, crop=crop)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return pred.__dict__


@app.get("/api/graph")
def graph_preview(crop: str, date: str, window: int = 7):
    """Edges of the spatial graph among farms growing a crop, for the map."""
    from ..graph.build import site_distance_matrix
    from ..data.farms import SITES

    svc = service()
    sites = [s for s in SITES if crop in s.crops]
    if len(sites) < 2:
        return {"nodes": [], "edges": []}

    sub = to_frame(tuple(sites))
    dist, bearing, ids = site_distance_matrix(sub)

    radius = _cfg.graph.radius_km
    k = _cfg.graph.k_neighbours
    min_nb = _cfg.graph.min_neighbours

    edges = []
    for i, sid in enumerate(ids):
        order = [j for j in dist[i].argsort() if j != i]
        within = [j for j in order if dist[i, j] <= radius][:k]
        if len(within) < min_nb:
            within = order[:min_nb]
        for j in within:
            edges.append({
                "source": sid, "target": ids[j],
                "distance_km": round(float(dist[i, j]), 1),
                "bearing_deg": round(float(bearing[i, j]), 1),
            })

    risk = {r["site_id"]: r for r in svc.regional_forecast(Date.fromisoformat(date), crop)}
    nodes = [
        {**r, **risk.get(r["site_id"], {})}
        for r in sub[["site_id", "name", "lat", "lon", "state"]].to_dict("records")
    ]
    return {"crop": crop, "date": date, "nodes": nodes, "edges": edges}


# --- literature survey + live demonstrations -------------------------------
@app.get("/api/literature")
def survey():
    """The 18-paper related-work table, each row tied to a runnable demo."""
    return {"papers": literature.as_records(), "demos": literature.demo_catalogue()}


@app.get("/api/demo/{demo_id}")
def demo(demo_id: str, crop: str = "Tomato", date: str = "2023-07-15"):
    """Run one literature-gap demonstration against the live system."""
    # Most demonstrations are properties of the data and the physics, not of the
    # learned weights, so they are served without a checkpoint. Only the ones
    # that actually run the network need the trained model.
    if demo_id in ("backbone_benchmark", "ablation", "missing_data"):
        return demos.demo_table(_cfg, demo_id)

    light = demos.ClimateOnlyService(_cfg)
    try:
        if demo_id == "climate_niche":
            return demos.demo_climate_niche(_cfg, light)
        if demo_id == "physics_constraints":
            return demos.demo_physics(_cfg, light)
        if demo_id == "graph_structure":
            return demos.demo_graph_structure(_cfg, light)
        if demo_id == "epidemiology":
            return demos.demo_epidemiology(_cfg, light)
        if demo_id == "contagion":
            return demos.demo_contagion(_cfg, light, crop)
        if demo_id == "multi_horizon":
            return demos.demo_multi_horizon(_cfg, light, crop, date)
        if demo_id == "risk_map":
            return demos.demo_risk_map(_cfg, light, crop, date)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not built yet - run scripts/01_build_dataset.py ({exc})",
        ) from exc

    # node_features reports the trained model's actual dimensions.
    try:
        return demos.run_demo(demo_id, _cfg, service())
    except KeyError as exc:
        raise HTTPException(status_code=404,
                            detail=f"Unknown demo {demo_id!r}") from exc
