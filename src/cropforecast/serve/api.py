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

from ..config import load_config
from ..data.farms import to_frame
from . import demos, literature
from .inference import ForecastService

WEB_DIR = Path(__file__).parent / "web"

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
