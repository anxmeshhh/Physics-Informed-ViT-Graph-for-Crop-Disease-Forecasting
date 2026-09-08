"""Inference service: one leaf image + a farm + a date -> a forecast.

Holds the trained artefacts in memory and runs the full path end to end:

    image -> frozen backbone -> fusion (+ climate, + metadata) -> GNN -> heads

Single-image inference has no neighbours to pass messages between, so the graph
encoder runs on a one-node graph with no edges. That is the honest behaviour: a
GNN given no neighbourhood falls back to its self-transform. When several
observations are submitted together the real graph is built over them.
"""
from __future__ import annotations

import io
import json
import pickle
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..data.farms import CROP_SEASONS, SITES, site_index, to_frame
from ..features.tabular import META_CATEGORICAL, META_NUMERIC, _season_features
from ..models.backbones import load_backbone
from ..models.full_model import CropDiseaseForecastNet
from ..physics.epidemiology import PROFILES, crop_pressure
from ..physics.losses import PHYSICS_COLUMNS

RISK_LABELS = ("Low", "Medium", "High")


@dataclass
class Prediction:
    disease: str
    crop: str
    confidence: float
    top_k: list[dict]
    horizons: list[int]
    risk: list[float]
    risk_band: list[str]
    risk_confidence: list[float]
    uncertainty: list[float]
    climate: dict
    site: dict
    date: str
    advisory: str


class ForecastService:
    """Loads every trained artefact once and serves predictions."""

    def __init__(self, cfg, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt_dir = Path(cfg.paths.checkpoints)
        processed = Path(cfg.paths.processed)

        blob = torch.load(ckpt_dir / "final_model.pt", map_location=self.device)
        self.meta = blob["meta"]
        self.model_cfg = blob["config"]
        self.class_names: list[str] = json.loads(
            (ckpt_dir / "class_names.json").read_text(encoding="utf-8")
        )
        with open(ckpt_dir / "feature_space.pkl", "rb") as fh:
            self.space = pickle.load(fh)

        self.model = CropDiseaseForecastNet(
            vision_dim=self.meta["vision_dim"],
            climate_dim=self.meta["climate_dim"],
            meta_dim=self.meta["meta_dim"],
            num_classes=self.meta["num_classes"],
            horizons=tuple(self.meta["horizons"]),
            fusion_dim=self.model_cfg["fusion_dim"],
            gnn_conv=self.model_cfg["gnn_conv"],
            gnn_hidden=self.model_cfg["gnn_hidden"],
            gnn_layers=self.model_cfg["gnn_layers"],
            gnn_heads=self.model_cfg["gnn_heads"],
        ).to(self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

        self.backbone = load_backbone(
            self.model_cfg["backbone"], frozen=True, output_attentions=True
        ).to(self.device).eval()

        from transformers import AutoImageProcessor
        from ..models.backbones import SPECS
        proc = AutoImageProcessor.from_pretrained(SPECS[self.model_cfg["backbone"]].hf_id)
        self.norm_mean = torch.tensor(proc.image_mean, device=self.device).view(1, 3, 1, 1)
        self.norm_std = torch.tensor(proc.image_std, device=self.device).view(1, 3, 1, 1)

        self.climate = pd.read_parquet(processed / "climate_features.parquet")
        self.climate["date"] = pd.to_datetime(self.climate["date"])
        self.sites_df = to_frame()
        self.site_lookup = site_index()
        self.horizons = list(self.meta["horizons"])

    # -- helpers ------------------------------------------------------------
    def available_dates(self, site_id: str) -> tuple[str, str]:
        sub = self.climate[self.climate.site_id == site_id]
        return str(sub.date.min().date()), str(sub.date.max().date())

    def climate_row(self, site_id: str, when: Date) -> pd.DataFrame:
        mask = (self.climate.site_id == site_id) & \
               (self.climate.date == pd.Timestamp(when))
        row = self.climate[mask]
        if row.empty:
            raise ValueError(
                f"No climate record for {site_id} on {when}. "
                f"Available range: {self.available_dates(site_id)}"
            )
        return row.reset_index(drop=True)

    def _embed(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device).float() / 255.0
        x = (x - self.norm_mean) / self.norm_std
        with torch.no_grad():
            return self.backbone(x)["pooled"]

    def _tabular(self, row: pd.DataFrame, crop: str, when: Date):
        site = self.site_lookup[row.site_id.iloc[0]]
        frame = row.copy()
        frame["crop"] = crop
        frame["soil_type"] = site.soil_type
        frame["agro_zone"] = site.agro_zone

        climate = self.space.scaler_climate.transform(
            frame[self.space.climate_cols].to_numpy(dtype=np.float64)
        ).astype(np.float32)

        meta = np.concatenate([
            self.space.scaler_meta.transform(
                frame[META_NUMERIC].to_numpy(dtype=np.float64)),
            _season_features(frame.date),
            self.space.encoder_meta.transform(frame[META_CATEGORICAL]),
        ], axis=1).astype(np.float32)

        return (torch.from_numpy(np.nan_to_num(climate)).to(self.device),
                torch.from_numpy(np.nan_to_num(meta)).to(self.device))

    # -- main entry point ---------------------------------------------------
    @torch.no_grad()
    def predict(self, image: Image.Image, site_id: str, when: Date,
                crop: str | None = None, top_k: int = 5) -> Prediction:
        if site_id not in self.site_lookup:
            raise ValueError(f"Unknown site {site_id!r}")
        site = self.site_lookup[site_id]

        row = self.climate_row(site_id, when)
        vision = self._embed(image)

        # Crop is needed for the metadata encoding. If the caller does not say,
        # take the first crop this farm grows and correct it after the image has
        # been classified.
        crop_guess = crop or site.crops[0]
        climate_t, meta_t = self._tabular(row, crop_guess, when)

        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        out = self.model(vision, climate_t, meta_t, empty_edges, None)

        probs = torch.softmax(out["class_logits"], dim=-1)[0]
        order = probs.argsort(descending=True)[:top_k]
        top = [{"class": self.class_names[int(i)],
                "probability": float(probs[int(i)])} for i in order]
        best = self.class_names[int(order[0])]
        best_crop = best.split("___", 1)[0]

        # Re-run with the crop the image actually shows, so the metadata stream
        # is consistent with the diagnosis.
        if crop is None and best_crop != crop_guess and best_crop in {
            c for s in SITES for c in s.crops
        }:
            climate_t, meta_t = self._tabular(row, best_crop, when)
            out = self.model(vision, climate_t, meta_t, empty_edges, None)

        risk = out["risk"][0].tolist()
        levels = out["level_logits"][0].softmax(-1)
        bands = [RISK_LABELS[int(l.argmax())] for l in levels]
        band_conf = [float(l.max()) for l in levels]
        sigma = (out["logvar"][0] * 0.5).exp().tolist()

        clim = row.iloc[0]
        climate_summary = {
            "temperature_C": round(float(clim.temperature_2m_mean), 1),
            "humidity_pct": round(float(clim.relative_humidity_2m_mean), 1),
            "rainfall_mm": round(float(clim.precipitation_sum), 1),
            "wind_kmh": round(float(clim.wind_speed_10m_max), 1),
            "leaf_wetness_h": round(float(clim.leaf_wetness_hours), 1),
            "vpd_kPa": round(float(clim.vpd_kpa), 2),
            "gdd": round(float(clim.gdd), 1),
        }

        return Prediction(
            disease=best,
            crop=best_crop,
            confidence=float(probs[int(order[0])]),
            top_k=top,
            horizons=self.horizons,
            risk=[round(r, 4) for r in risk],
            risk_band=bands,
            risk_confidence=[round(c, 4) for c in band_conf],
            uncertainty=[round(s, 4) for s in sigma],
            climate=climate_summary,
            site={"site_id": site.site_id, "name": site.name, "state": site.state,
                  "lat": site.lat, "lon": site.lon, "soil": site.soil_type,
                  "agro_zone": site.agro_zone},
            date=str(when),
            advisory=self.advisory(best, bands, climate_summary),
        )

    # -- agronomic advisory -------------------------------------------------
    def advisory(self, disease: str, bands: list[str], climate: dict) -> str:
        """A short, plain-language action note derived from the forecast."""
        if disease.endswith("___healthy"):
            base = "No disease detected on this leaf."
        else:
            pretty = disease.split("___", 1)[1].replace("_", " ")
            base = f"{pretty} identified."

        worst = max(bands, key=lambda b: RISK_LABELS.index(b))
        profile = PROFILES.get(disease)

        if worst == "High":
            when = bands.index("High")
            days = self.horizons[when]
            note = (f" Conditions turn highly favourable within {days} day"
                    f"{'s' if days > 1 else ''}.")
            if profile and profile.moisture == "wet":
                note += (f" Leaf wetness is {climate['leaf_wetness_h']} h - "
                         "schedule a protectant spray before the next wet period "
                         "and avoid overhead irrigation.")
            elif profile and profile.moisture == "humid_no_free_water":
                note += " Improve canopy airflow; free water suppresses this pathogen."
            elif profile and profile.moisture == "splash":
                note += " Rain splash is the main spread route - avoid working the field wet."
            elif profile and profile.moisture == "dry_hot":
                note += " Hot dry spell favours this pest; scout leaf undersides."
            else:
                note += " Increase scouting frequency."
        elif worst == "Medium":
            note = " Moderate risk ahead - keep scouting on the normal schedule."
        else:
            note = " Conditions stay unfavourable for infection over the forecast window."
        return base + note

    # -- regional view ------------------------------------------------------
    def regional_forecast(self, when: Date, crop: str) -> list[dict]:
        """Physics-model risk for every farm growing ``crop`` on a date.

        This drives the map. It is the agronomic pressure computed from the real
        recorded weather, which is exactly the quantity the network is trained to
        forecast, and it is defined for every farm rather than only the ones with
        an uploaded photograph.
        """
        sites = [s for s in SITES if crop in s.crops]
        rows = []
        for s in sites:
            sub = self.climate[
                (self.climate.site_id == s.site_id)
                & (self.climate.date == pd.Timestamp(when))
            ]
            if sub.empty:
                continue
            pressure = float(crop_pressure(crop, sub)[0])
            band = ("High" if pressure > 0.55 else
                    "Medium" if pressure > 0.25 else "Low")
            rows.append({
                "site_id": s.site_id, "name": s.name, "state": s.state,
                "lat": s.lat, "lon": s.lon, "risk": round(pressure, 4),
                "band": band,
                "temperature_C": round(float(sub.temperature_2m_mean.iloc[0]), 1),
                "humidity_pct": round(float(sub.relative_humidity_2m_mean.iloc[0]), 1),
                "rainfall_mm": round(float(sub.precipitation_sum.iloc[0]), 1),
                "leaf_wetness_h": round(float(sub.leaf_wetness_hours.iloc[0]), 1),
            })
        return sorted(rows, key=lambda r: -r["risk"])

    def climate_series(self, site_id: str, start: Date, end: Date) -> dict:
        sub = self.climate[
            (self.climate.site_id == site_id)
            & (self.climate.date >= pd.Timestamp(start))
            & (self.climate.date <= pd.Timestamp(end))
        ].sort_values("date")
        return {
            "dates": [str(d.date()) for d in sub.date],
            "temperature": sub.temperature_2m_mean.round(1).tolist(),
            "humidity": sub.relative_humidity_2m_mean.round(1).tolist(),
            "rainfall": sub.precipitation_sum.round(1).tolist(),
            "leaf_wetness": sub.leaf_wetness_hours.round(1).tolist(),
            "vpd": sub.vpd_kpa.round(2).tolist(),
        }

    def crops(self) -> list[str]:
        return sorted({c for s in SITES for c in s.crops})

    def season_for(self, crop: str) -> dict:
        (m, d), length = CROP_SEASONS[crop]
        return {"start_month": m, "start_day": d, "length_days": length}
