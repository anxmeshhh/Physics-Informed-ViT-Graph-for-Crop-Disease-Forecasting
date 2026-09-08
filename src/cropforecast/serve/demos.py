"""Live demonstrations backing each literature-survey research gap.

Every function returns plain JSON-serialisable data computed from the *actual*
trained system and the *actual* cached data - never a canned example. That is
the point: a reviewer reading "this paper does not model inter-field spread" can
press a button and watch the graph being built.
"""
from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.farms import SITES, to_frame
from ..graph.build import build_graph, site_distance_matrix
from ..physics.epidemiology import (
    PROFILES, favourability, temperature_response,
)
from ..physics.contagion import (
    EpidemicParams, dispersal_kernel, simulate_crop, spatial_autocorrelation,
)
from ..physics.losses import (
    diffusion_loss, dry_suppression_loss, monotonic_horizon_loss,
)


class ClimateOnlyService:
    """The parts of the service that need data but not a trained network.

    Several demonstrations - the pathogen response curves, the regional risk
    map, the epidemic - are properties of the *data and the physics*, not of the
    learned weights. Requiring a checkpoint for those would mean a team could not
    show anything until training finished, so they are served from this instead.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._climate = None
        self._outbreaks = None
        self.horizons = list(cfg.model.heads.horizons)

    @property
    def climate(self) -> pd.DataFrame:
        if self._climate is None:
            df = pd.read_parquet(
                Path(self.cfg.paths.processed) / "climate_features.parquet")
            df["date"] = pd.to_datetime(df["date"])
            self._climate = df
        return self._climate

    @property
    def outbreaks(self) -> pd.DataFrame | None:
        if self._outbreaks is None:
            path = Path(self.cfg.paths.processed) / "outbreaks.parquet"
            if not path.exists():
                return None
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            self._outbreaks = df
        return self._outbreaks

    def regional_forecast(self, when: Date, crop: str) -> list[dict]:
        """Risk at every farm growing this crop, from real weather + epidemic."""
        from ..physics.epidemiology import crop_pressure
        from ..physics.contagion import combined_risk

        sites = [s for s in SITES if crop in s.crops]
        outbreaks = self.outbreaks
        rows = []
        for s in sites:
            sub = self.climate[(self.climate.site_id == s.site_id)
                               & (self.climate.date == pd.Timestamp(when))]
            if sub.empty:
                continue
            pressure = float(crop_pressure(crop, sub)[0])
            inoc = 0.0
            if outbreaks is not None:
                hit = outbreaks[(outbreaks.crop == crop)
                                & (outbreaks.site_id == s.site_id)
                                & (outbreaks.date == pd.Timestamp(when))]
                if len(hit):
                    inoc = float(hit.infection_pressure.iloc[0])
                pressure = float(combined_risk(np.array([pressure]),
                                               np.array([inoc]))[0])
            band = "High" if pressure > 0.55 else "Medium" if pressure > 0.25 else "Low"
            rows.append({
                "site_id": s.site_id, "name": s.name, "state": s.state,
                "lat": s.lat, "lon": s.lon, "risk": round(pressure, 4),
                "band": band, "inoculum": round(inoc, 4),
                "temperature_C": round(float(sub.temperature_2m_mean.iloc[0]), 1),
                "humidity_pct": round(float(sub.relative_humidity_2m_mean.iloc[0]), 1),
                "rainfall_mm": round(float(sub.precipitation_sum.iloc[0]), 1),
                "leaf_wetness_h": round(float(sub.leaf_wetness_hours.iloc[0]), 1),
            })
        return sorted(rows, key=lambda r: -r["risk"])


def demo_contagion(cfg, service, crop: str = "Tomato") -> dict:
    """Show that a farm's future state depends on its neighbours, not just itself.

    This is the measurement that justifies the graph. If neighbours add nothing
    beyond a farm's own history, no amount of message passing can help.
    """
    svc = service if isinstance(service, ClimateOnlyService) else ClimateOnlyService(cfg)
    sim = svc.outbreaks
    if sim is not None:
        sim = sim[sim.crop == crop]
    if sim is None or sim.empty:
        sim = simulate_crop(svc.climate, crop)

    stats = [spatial_autocorrelation(sim, svc.climate, crop, lag_days=h)
             for h in (1, 3, 7)]

    # A worked example: the largest outbreak, and where it came from.
    wide = sim.pivot(index="date", columns="site_id", values="infection_pressure")
    peak_idx = np.unravel_index(np.nanargmax(wide.to_numpy()), wide.shape)
    peak_date = wide.index[peak_idx[0]]
    peak_site = wide.columns[peak_idx[1]]
    lead = wide.loc[:peak_date].tail(8)

    timeline = []
    for d, row in lead.iterrows():
        top = row.sort_values(ascending=False).head(3)
        timeline.append({
            "date": str(d.date()),
            "focus_site": peak_site,
            "focus_pressure": round(float(row.get(peak_site, 0.0)), 4),
            "highest_elsewhere": ", ".join(
                f"{sid} {v:.2f}" for sid, v in top.items() if sid != peak_site),
        })

    params = EpidemicParams()
    return {
        "kind": "contagion",
        "crop": crop,
        "params": {"epsilon": params.epsilon, "beta": params.beta,
                   "gamma": params.gamma, "recovery": params.recovery,
                   "length_scale_km": params.length_scale_km,
                   "wind_boost": params.wind_boost},
        "autocorrelation": stats,
        "peak": {"site_id": peak_site, "date": str(peak_date.date()),
                 "pressure": round(float(wide.to_numpy()[peak_idx]), 4)},
        "timeline": timeline,
        "note": ("Between-farm transmission (gamma) dominates local build-up "
                 "(beta), so outbreaks arrive from elsewhere. That is what gives "
                 "the graph something to learn: a farm's risk in three days "
                 "depends on what its neighbours are carrying today."),
    }


def demo_graph_structure(cfg, service) -> dict:
    """Build the real spatio-temporal graph on a slice of observations."""
    obs = pd.read_parquet(Path(cfg.paths.processed) / "observations.parquet")
    sample = obs.sample(min(3000, len(obs)), random_state=cfg.project.seed) \
                .reset_index(drop=True)
    g = build_graph(
        sample, to_frame(),
        k_neighbours=cfg.graph.k_neighbours, radius_km=cfg.graph.radius_km,
        time_window_days=cfg.graph.time_window_days,
        max_edges_per_node=cfg.graph.max_edges_per_node,
        wind_aware=cfg.graph.wind_aware,
        wind_alignment_threshold=cfg.graph.wind_alignment_threshold,
        min_neighbours=cfg.graph.min_neighbours,
        same_crop=cfg.graph.same_crop, seed=cfg.project.seed,
    )
    summary = g.summary()

    # A few concrete edges, so the abstraction is visible.
    examples = []
    for e in range(min(8, g.edge_index.shape[1])):
        s, t = int(g.edge_index[0, e]), int(g.edge_index[1, e])
        examples.append({
            "from_site": sample.site_id.iloc[s],
            "from_date": str(sample.date.iloc[s])[:10],
            "to_site": sample.site_id.iloc[t],
            "to_date": str(sample.date.iloc[t])[:10],
            "crop": sample.crop.iloc[s],
            "distance_norm": round(float(g.edge_attr[e, 0]), 3),
            "days_apart": int(round(float(g.edge_attr[e, 1]) * cfg.graph.time_window_days)),
            "wind_alignment": round(float(g.edge_attr[e, 2]), 3),
            "downwind": bool(g.edge_attr[e, 3]),
        })

    return {
        "kind": "graph",
        "summary": summary,
        "settings": {
            "k_neighbours": cfg.graph.k_neighbours,
            "radius_km": cfg.graph.radius_km,
            "time_window_days": cfg.graph.time_window_days,
            "same_crop_only": cfg.graph.same_crop,
        },
        "examples": examples,
        "note": (f"{summary['downwind_edges']:,} of {summary['num_edges']:,} edges "
                 "are directed downwind edges, added only where the wind measured "
                 "at the source farm was actually blowing toward the target."),
    }


def demo_physics(cfg, service) -> dict:
    """Evaluate each physics constraint on legal and illegal behaviour."""
    torch.manual_seed(cfg.project.seed)
    ei = torch.randint(0, 64, (2, 300))
    ew = torch.rand(300)

    increasing = torch.arange(4).float().repeat(16, 1)
    decreasing = -increasing
    uniform = torch.ones(64, 4)
    noisy = torch.rand(64, 4)
    high = torch.full((32, 4), 0.9)

    checks = [
        {"constraint": "Uncertainty must not shrink with horizon",
         "legal_case": "uncertainty grows with horizon",
         "legal_penalty": round(float(monotonic_horizon_loss(increasing)), 5),
         "illegal_case": "uncertainty shrinks with horizon",
         "illegal_penalty": round(float(monotonic_horizon_loss(decreasing)), 5)},
        {"constraint": "Risk must be smooth across connected farms",
         "legal_case": "neighbouring farms share risk",
         "legal_penalty": round(float(diffusion_loss(uniform, ei, ew)), 5),
         "illegal_case": "risk jumps discontinuously between neighbours",
         "illegal_penalty": round(float(diffusion_loss(noisy, ei, ew)), 5)},
        {"constraint": "Dry air suppresses fungal risk",
         "legal_case": "high risk under humid air (VPD 0.3 kPa)",
         "legal_penalty": round(float(dry_suppression_loss(high, torch.full((32,), 0.3))), 5),
         "illegal_case": "high risk under strongly drying air (VPD 2.0 kPa)",
         "illegal_penalty": round(float(dry_suppression_loss(high, torch.full((32,), 2.0))), 5)},
    ]
    return {
        "kind": "physics",
        "checks": checks,
        "weights": dict(cfg.physics.weights),
        "lambda": cfg.physics.lambda_physics,
        "note": ("The loss never sees the epidemiological model used to build the "
                 "benchmark. It states only directional facts, so the network "
                 "must still learn the quantitative relationship from data."),
    }


def demo_epidemiology(cfg, service) -> dict:
    """Pathogen responses evaluated on real recorded weather."""
    climate = service.climate
    site = "HP-SML"
    sub = climate[climate.site_id == site]
    wettest = sub.loc[[sub.leaf_wetness_hours.idxmax()]]
    driest = sub.loc[[sub.vpd_kpa.idxmax()]]

    classes = ["Potato___Late_blight", "Apple___Apple_scab",
               "Squash___Powdery_mildew",
               "Tomato___Spider_mites Two-spotted_spider_mite",
               "Tomato___Bacterial_spot"]

    def describe(row):
        r = row.iloc[0]
        return {
            "date": str(r.date)[:10],
            "temperature_C": round(float(r.temperature_2m_mean), 1),
            "humidity_pct": round(float(r.relative_humidity_2m_mean), 0),
            "rainfall_mm": round(float(r.precipitation_sum), 1),
            "leaf_wetness_h": round(float(r.leaf_wetness_hours), 1),
            "vpd_kPa": round(float(r.vpd_kpa), 2),
        }

    rows = []
    for cls in classes:
        p = PROFILES[cls]
        rows.append({
            "disease": cls,
            "t_opt": p.t_opt,
            "moisture_mode": p.moisture,
            "note": p.note,
            "favourability_wet_day": round(float(favourability(cls, wettest)[0]), 3),
            "favourability_dry_day": round(float(favourability(cls, driest)[0]), 3),
        })

    # Response curves for plotting.
    temps = np.linspace(0, 42, 90)
    curves = {
        cls: [round(float(v), 4) for v in temperature_response(temps, PROFILES[cls])]
        for cls in classes
    }

    return {
        "kind": "epidemiology",
        "site": "Shimla (HP-SML)",
        "wet_day": describe(wettest),
        "dry_day": describe(driest),
        "rows": rows,
        "temps": [round(float(t), 2) for t in temps],
        "curves": curves,
        "note": ("Powdery mildew collapses on the wet day because free water "
                 "bursts its conidia, while late blight peaks. Spider mites do "
                 "the reverse. These opposite responses are what make the "
                 "climate branch informative."),
    }


def demo_climate_niche(cfg, service) -> dict:
    """Mean weather at which each class was observed."""
    obs = pd.read_parquet(
        Path(cfg.paths.processed) / "observations.parquet",
        columns=["class_name", "crop", "image_path", "temperature_2m_mean",
                 "relative_humidity_2m_mean", "leaf_wetness_hours", "vpd_kpa"],
    )
    agg = (obs.groupby("class_name")
           .agg(n=("image_path", "size"),
                temperature_C=("temperature_2m_mean", "mean"),
                humidity_pct=("relative_humidity_2m_mean", "mean"),
                leaf_wetness_h=("leaf_wetness_hours", "mean"),
                vpd_kPa=("vpd_kpa", "mean"))
           .round(2).reset_index().sort_values("temperature_C"))
    agg["healthy"] = agg.class_name.str.endswith("___healthy")
    return {
        "kind": "niche",
        "rows": agg.to_dict(orient="records"),
        "note": ("Cool, wet classes sit at the top; hot, dry classes at the "
                 "bottom. Nothing enforced this - it follows from placing each "
                 "leaf where the real recorded weather suited its pathogen."),
    }


def demo_multi_horizon(cfg, service, crop: str = "Tomato",
                       date: str = "2023-07-15") -> dict:
    """Physics-model risk trajectory across the forecast horizons."""
    when = Date.fromisoformat(date)
    regional = service.regional_forecast(when, crop)
    if not regional:
        return {"kind": "forecast", "rows": [], "note": "No data for that date."}

    top = regional[0]
    from ..physics.epidemiology import crop_pressure
    trajectory = []
    for h in [0] + list(service.horizons):
        target = pd.Timestamp(when) + pd.Timedelta(days=h)
        sub = service.climate[(service.climate.site_id == top["site_id"])
                              & (service.climate.date == target)]
        if sub.empty:
            continue
        trajectory.append({
            "horizon_days": h,
            "date": str(target.date()),
            "risk": round(float(crop_pressure(crop, sub)[0]), 4),
            "temperature_C": round(float(sub.temperature_2m_mean.iloc[0]), 1),
            "humidity_pct": round(float(sub.relative_humidity_2m_mean.iloc[0]), 0),
            "leaf_wetness_h": round(float(sub.leaf_wetness_hours.iloc[0]), 1),
        })

    return {
        "kind": "forecast",
        "crop": crop, "date": date,
        "site": {"site_id": top["site_id"], "name": top["name"], "state": top["state"]},
        "trajectory": trajectory,
        "regional_top": regional[:8],
        "horizons": service.horizons,
        "note": ("Risk at horizon h is the agronomic pressure computed from the "
                 "weather that was actually recorded h days later, so the "
                 "forecast is scored against reality rather than a proxy."),
    }


def demo_risk_map(cfg, service, crop: str = "Tomato",
                  date: str = "2023-07-15") -> dict:
    rows = service.regional_forecast(Date.fromisoformat(date), crop)
    return {
        "kind": "map", "crop": crop, "date": date, "sites": rows,
        "note": f"{len(rows)} farms grow {crop.replace('_', ' ')} in the registry.",
    }


def demo_node_features(cfg, service) -> dict:
    """What one graph node actually carries."""
    meta = service.meta
    return {
        "kind": "node",
        "streams": [
            {"stream": "Vision embedding",
             "dims": meta["vision_dim"],
             "source": f"frozen {service.model_cfg['backbone']} transformer",
             "example": "CLS token of the leaf photograph"},
            {"stream": "Climate features",
             "dims": meta["climate_dim"],
             "source": "Open-Meteo ERA5 + engineered agro-meteorology",
             "example": "VPD, leaf wetness hours, GDD, 3/7/14-day rolling means"},
            {"stream": "Farm metadata",
             "dims": meta["meta_dim"],
             "source": "registry + cyclical season encoding",
             "example": "latitude, longitude, altitude, soil type, agro-zone, season"},
        ],
        "fused_dim": cfg.model.fusion.hidden_dim,
        "note": ("Compare with graphs whose nodes encode only a disease location: "
                 f"each node here is a {meta['vision_dim']} + {meta['climate_dim']} "
                 f"+ {meta['meta_dim']} dimensional description of a real farm-day."),
    }


def demo_table(cfg, which: str) -> dict:
    """Serve a saved results table (benchmark or ablations)."""
    reports = Path(cfg.paths.reports)
    files = {
        "backbone_benchmark": ("stage3_backbone_benchmark.csv",
                               "Four transformer backbones, identical downstream pipeline"),
        "ablation": ("stage4_ablations.csv",
                     "Component ablations on the leaf-disjoint test set"),
        "missing_data": ("stage8_missing_data.csv",
                         "Graph value as observations become sparse"),
    }
    name, caption = files[which]
    path = reports / name
    if not path.exists():
        return {"kind": "table", "rows": [], "caption": caption,
                "note": f"Not generated yet - run the stage that writes {name}."}
    df = pd.read_csv(path).round(4)
    return {"kind": "table", "rows": df.to_dict(orient="records"),
            "columns": df.columns.tolist(), "caption": caption}


DISPATCH = {
    "contagion": demo_contagion,
    "graph_structure": demo_graph_structure,
    "physics_constraints": demo_physics,
    "epidemiology": demo_epidemiology,
    "climate_niche": demo_climate_niche,
    "multi_horizon": demo_multi_horizon,
    "risk_map": demo_risk_map,
    "node_features": demo_node_features,
}


def run_demo(demo_id: str, cfg, service) -> dict:
    if demo_id in ("backbone_benchmark", "ablation", "missing_data"):
        return demo_table(cfg, demo_id)
    if demo_id not in DISPATCH:
        raise KeyError(demo_id)
    return DISPATCH[demo_id](cfg, service)
