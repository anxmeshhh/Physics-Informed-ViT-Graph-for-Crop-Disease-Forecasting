"""Risk surface: the location assessment, evaluated on a grid across the region.

Thirty-seven monitored districts leave most of the map empty, and smearing
colour between them would invent values nobody measured. Instead every grid
cell is computed exactly as :mod:`serve.nearby` computes a clicked point: the
weather recorded at that cell, through the pathogen physics, plus disease
arriving from the districts around it on the wind. Clicking anywhere on the
heatmap and choosing *Near a location* therefore gives the same number.

The grid is 1 degree, limited to land within ``COVER_KM`` of a monitored
district - the region the network actually observes. Weather comes from the same
Open-Meteo archive as the districts, fetched in batches once per date window and
cached to disk, so a demo pays for each date only the first time.
"""
from __future__ import annotations

import time
from datetime import date as Date
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import PROJECT_ROOT
from ..data.climate import ARCHIVE_URL
from ..data.farms import SITES
from ..features.agromet import derive_daily_features
from ..graph.build import haversine_km, initial_bearing_deg
from ..physics.contagion import EpidemicParams, combined_risk, inoculum_term
from ..physics.epidemiology import crop_pressure

STEP = 1.0
LAT_RANGE = (6.0, 37.0)
LON_RANGE = (68.0, 97.0)
COVER_KM = 350.0
DAYS = 8                     # today and the coming week
BATCH = 100                  # locations per Open-Meteo request
# Only what the physics reads: 7 variables keeps each location at one API call.
GRID_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "relative_humidity_2m_max",
    "precipitation_sum", "wind_speed_10m_max",
]
CACHE_DIR = PROJECT_ROOT / "data" / "climate" / "grid"
PARAMS = EpidemicParams()


class WeatherUnavailable(RuntimeError):
    """The archive could not be reached or throttled us."""


@lru_cache(maxsize=1)
def grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lats, lons, covered) - the full lattice and which cells are in range."""
    lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-9, STEP)
    lons = np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-9, STEP)
    la = np.array([s.lat for s in SITES])
    lo = np.array([s.lon for s in SITES])
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    d = haversine_km(glat.ravel()[:, None], glon.ravel()[:, None], la[None, :], lo[None, :])
    covered = (d.min(axis=1) <= COVER_KM).reshape(glat.shape)
    return lats, lons, covered


def grid_weather(start: Date, end: Date) -> pd.DataFrame:
    """Daily weather for every covered cell over [start, end], cached to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    lats, lons, covered = grid()
    cells = [(i, j, lats[i], lons[j]) for i, j in zip(*np.nonzero(covered))]
    frames = []
    for k in range(0, len(cells), BATCH):
        chunk = cells[k:k + BATCH]
        payload = _fetch_batch(chunk, start, end)
        for (i, j, _, _), loc in zip(chunk, payload):
            df = pd.DataFrame(loc["daily"]).rename(columns={"time": "date"})
            df["i"], df["j"] = i, j
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = derive_daily_features(out)
    out.to_parquet(path, index=False)
    return out


def _fetch_batch(chunk: list, start: Date, end: Date, attempts: int = 3) -> list:
    """One batched archive request. A truncated or garbled reply is retried;
    a rate limit is not, since retrying straight away only extends it."""
    last = "no response"
    for attempt in range(attempts):
        try:
            resp = requests.get(ARCHIVE_URL, timeout=40, params={
                "latitude": ",".join(f"{c[2]:.2f}" for c in chunk),
                "longitude": ",".join(f"{c[3]:.2f}" for c in chunk),
                "start_date": str(start), "end_date": str(end),
                "daily": ",".join(GRID_VARS), "timezone": "Asia/Kolkata",
            })
            if resp.status_code == 429:
                raise WeatherUnavailable("the weather archive is rate-limiting; try again in a minute")
            if resp.ok:
                payload = resp.json()
                return payload if isinstance(payload, list) else [payload]
            last = f"weather archive returned {resp.status_code}"
        except WeatherUnavailable:
            raise
        except (requests.RequestException, ValueError) as exc:   # ValueError: bad JSON
            last = f"weather archive reply unreadable ({exc.__class__.__name__})"
        time.sleep(3 * (attempt + 1))
    raise WeatherUnavailable(last)


def build(climate: pd.DataFrame, outbreaks: pd.DataFrame | None,
          crop: str, when: Date) -> dict:
    end = when + timedelta(days=DAYS - 1)
    days = pd.date_range(when, end)
    lats, lons, covered = grid()
    w = grid_weather(when, end)

    # Weather here, per cell and day - crop_pressure is row-wise, so the whole
    # stacked frame goes through it at once.
    w = w[w.date.isin(days)].sort_values(["i", "j", "date"]).reset_index(drop=True)
    cell_ids = w[["i", "j"]].drop_duplicates().to_numpy()
    local = crop_pressure(crop, w).reshape(len(cell_ids), len(days))

    # Disease arriving, per cell and day: the contagion model's kernel from
    # every district growing this crop.
    sites = [s for s in SITES if crop in s.crops]
    ids = [s.site_id for s in sites]
    s_lat = np.array([s.lat for s in sites])
    s_lon = np.array([s.lon for s in sites])
    c_lat = lats[cell_ids[:, 0]]
    c_lon = lons[cell_ids[:, 1]]
    dist = haversine_km(c_lat[:, None], c_lon[:, None], s_lat[None, :], s_lon[None, :])
    to_cell = initial_bearing_deg(s_lat[None, :], s_lon[None, :], c_lat[:, None], c_lon[:, None])

    clim = climate[climate.site_id.isin(ids) & climate.date.isin(days)]
    piv = lambda col: (clim.pivot(index="date", columns="site_id", values=col)
                       .reindex(index=days, columns=ids).fillna(0.0).to_numpy())
    wind_dir, wind_spd = piv("wind_direction_10m_dominant"), piv("wind_speed_10m_max")
    if outbreaks is not None:
        ob = outbreaks[(outbreaks.crop == crop) & outbreaks.site_id.isin(ids)
                       & outbreaks.date.isin(days)]
        infected = (ob.pivot(index="date", columns="site_id", values="infection_pressure")
                    .reindex(index=days, columns=ids).fillna(0.0).to_numpy())
    else:
        infected = np.zeros((len(days), len(ids)))

    base = np.exp(-dist / PARAMS.length_scale_km)                       # (cells, sites)
    incoming = np.zeros_like(local)
    for t in range(len(days)):
        align = np.cos(np.radians((wind_dir[t] + 180.0) % 360.0 - to_cell))
        speed = np.clip(wind_spd[t] / 30.0, 0.0, 1.5)
        kernel = base * (1.0 + PARAMS.wind_boost * np.clip(align, 0.0, None) * speed)
        incoming[:, t] = PARAMS.gamma * (kernel * infected[t]).sum(axis=1)

    risk = combined_risk(local, incoming)
    spread = inoculum_term(incoming)

    # Each district's own reading, so the surface can be pinned to it: a 1 degree
    # cell sits up to ~70 km from a district, and a local shower can make the two
    # genuinely differ. These are exactly the values behind the bulletin's alerts.
    anchors = []
    for s in sites:
        sub = climate[(climate.site_id == s.site_id) & climate.date.isin(days)].sort_values("date")
        if len(sub) != len(days):
            continue
        own = crop_pressure(crop, sub)
        anchors.append({"name": s.name, "lat": s.lat, "lon": s.lon,
                        "today": round(float(own[0]), 3), "week": round(float(own.max()), 3)})

    def lattice(values: np.ndarray) -> list[list[float | None]]:
        out = np.full(covered.shape, np.nan)
        out[cell_ids[:, 0], cell_ids[:, 1]] = values
        return [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in out]

    return {
        "crop": crop, "date": str(when), "until": str(end),
        "step": STEP, "lats": lats.round(3).tolist(), "lons": lons.round(3).tolist(),
        "cells": int(len(cell_ids)),
        "anchors": anchors,
        "layers": {
            "weather": {"today": lattice(local[:, 0]), "week": lattice(local.max(axis=1))},
            "spread": {"today": lattice(spread[:, 0]), "week": lattice(spread.max(axis=1))},
            "risk": {"today": lattice(risk[:, 0]), "week": lattice(risk.max(axis=1))},
        },
    }
