"""Risk at one precise location, and where it could arrive from.

The bulletin judges each monitored district on its own weather. A farmer or an
officer standing somewhere else asks two different questions: is the weather
*here* favourable to disease, and is disease *arriving* from the farms around
me? Those are the two halves of the project's forecasting target
(:func:`physics.contagion.combined_risk`), so this module answers them with the
same pieces the training data was built from:

* **Conduciveness here** - the weather recorded at these exact coordinates
  (Open-Meteo archive, the same source as every district), through the same
  derived agro-meteorology and pathogen physics.
* **Inoculum arriving** - the simulated epidemic's infection pressure at every
  district growing the crop, carried here by the contagion model's dispersal
  kernel: exponential decay with distance, boosted when the wind at the source
  was actually blowing this way.

Both are computed day by day across the past week and the coming one, so the
location gets the same alert, trend, threat and actions as a district.
"""
from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from ..data.climate import ARCHIVE_URL, DAILY_VARS
from ..data.farms import SITES
from ..features.agromet import derive_daily_features
from ..graph.build import haversine_km, initial_bearing_deg
from ..physics.contagion import EpidemicParams, combined_risk, inoculum_term
from ..physics.epidemiology import PROFILES, crop_pressure
from . import bulletin

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
INDIA = (6.0, 37.5, 68.0, 97.5)          # lat_min, lat_max, lon_min, lon_max
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
PARAMS = EpidemicParams()


def compass(deg: float) -> str:
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


@lru_cache(maxsize=256)
def _point_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Daily archive weather at a point. Cached: a demo clicks the same spots."""
    resp = requests.get(ARCHIVE_URL, timeout=12, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS), "timezone": "auto",
    })
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return derive_daily_features(df)


def geocode(query: str) -> list[dict]:
    """Place-name search, Indian results first. No key needed."""
    resp = requests.get(GEOCODE_URL, timeout=8, params={
        "name": query, "count": 8, "language": "en", "format": "json"})
    resp.raise_for_status()
    hits = resp.json().get("results") or []
    hits.sort(key=lambda h: h.get("country_code") != "IN")
    return [{
        "name": h["name"],
        "region": ", ".join(x for x in (h.get("admin2"), h.get("admin1"), h.get("country")) if x),
        "lat": round(h["latitude"], 4), "lon": round(h["longitude"], 4),
    } for h in hits]


def assess_location(cfg, climate: pd.DataFrame, outbreaks: pd.DataFrame | None,
                    crop: str, when: Date, lat: float, lon: float,
                    radius_km: float = 300.0) -> dict:
    start = when - timedelta(days=bulletin.TREND_DAYS - 1)
    end = when + timedelta(days=bulletin.OUTLOOK_DAYS)
    today = pd.Timestamp(when)
    days = pd.date_range(start, end)

    sites = [s for s in SITES if crop in s.crops]
    if not sites:
        raise ValueError(f"No monitored district grows {crop}")
    lats = np.array([s.lat for s in sites])
    lons = np.array([s.lon for s in sites])
    dist = haversine_km(lats, lons, lat, lon)                 # source -> here
    to_here = initial_bearing_deg(lats, lons, lat, lon)       # bearing source -> here
    from_here = initial_bearing_deg(lat, lon, lats, lons)     # for "65 km NE"
    nearest = int(np.argmin(dist))

    # --- conduciveness here ------------------------------------------------
    note = None
    try:
        weather = _point_weather(round(lat, 3), round(lon, 3), str(start), str(end))
        source = "Weather recorded at this exact location (Open-Meteo archive)"
    except Exception:                                          # noqa: BLE001
        # Offline or throttled: fall back to the nearest monitored district,
        # and say so rather than pretend to precision we do not have.
        s = sites[nearest]
        weather = climate[(climate.site_id == s.site_id)
                          & climate.date.between(pd.Timestamp(start), pd.Timestamp(end))]
        source = f"Weather from the nearest monitored district, {s.name}"
        note = ("The live weather lookup was unavailable, so conditions here are "
                f"approximated by {s.name}, {dist[nearest]:.0f} km away.")
    weather = weather.set_index("date").reindex(days).ffill().bfill().rename_axis("date").reset_index()
    local = crop_pressure(crop, weather)

    # --- inoculum arriving ---------------------------------------------------
    ids = [s.site_id for s in sites]
    clim = climate[climate.site_id.isin(ids) & climate.date.isin(days)]
    wind_dir = clim.pivot(index="date", columns="site_id",
                          values="wind_direction_10m_dominant").reindex(index=days, columns=ids)
    wind_spd = clim.pivot(index="date", columns="site_id",
                          values="wind_speed_10m_max").reindex(index=days, columns=ids)
    if outbreaks is not None:
        ob = outbreaks[(outbreaks.crop == crop) & outbreaks.site_id.isin(ids)
                       & outbreaks.date.isin(days)]
        infected = ob.pivot(index="date", columns="site_id",
                            values="infection_pressure").reindex(index=days, columns=ids)
    else:
        infected = pd.DataFrame(0.0, index=days, columns=ids)
    wind_dir, wind_spd, infected = (x.fillna(0.0).to_numpy() for x in (wind_dir, wind_spd, infected))

    # Same kernel as the simulation: spores leave the source travelling toward
    # wind_dir + 180, and arrive here when that points along source -> here.
    base = np.exp(-dist / PARAMS.length_scale_km)
    align = np.cos(np.radians((wind_dir + 180.0) % 360.0 - to_here[None, :]))
    speed = np.clip(wind_spd / 30.0, 0.0, 1.5)
    kernel = base[None, :] * (1.0 + PARAMS.wind_boost * np.clip(align, 0.0, None) * speed)
    contrib = PARAMS.gamma * kernel * infected                # (days, sites)
    imported = contrib.sum(axis=1)

    risk = combined_risk(local, imported)
    peers = [c for c in PROFILES if c.split("___", 1)[0] == crop]
    here = bulletin.assess(risk, weather, today, peers)
    i_today = list(days).index(today)

    # --- where it could come from -------------------------------------------
    total = float(contrib[i_today].sum())
    district_alert = {r["site_id"]: r for r in bulletin.build(
        cfg, climate, crop, when)["districts"]}
    sources = []
    for j in np.argsort(-contrib[i_today]):
        if dist[j] > radius_km and j != nearest:
            continue
        s = sites[j]
        a = float(align[i_today, j])
        sources.append({
            "site_id": s.site_id, "name": s.name, "state": s.state,
            "lat": s.lat, "lon": s.lon,
            "distance_km": round(float(dist[j]), 1),
            "direction": compass(from_here[j]),
            "wind_toward_you": bool(a > 0.5 and wind_spd[i_today, j] >= 5),
            "wind_kmh": round(float(wind_spd[i_today, j]), 1),
            "infection": round(float(inoculum_term(infected[i_today, j])), 4),
            "share": round(float(contrib[i_today, j]) / total, 4) if total > 0 else 0.0,
            "alert": district_alert.get(s.site_id, {}).get("alert", "Low"),
        })

    in_india = INDIA[0] <= lat <= INDIA[1] and INDIA[2] <= lon <= INDIA[3]
    return {
        "lat": lat, "lon": lon, "crop": crop, "crop_label": bulletin.crop_label(crop),
        "date": str(when), "outlook_until": str(end), "radius_km": radius_km,
        "nearest": {"name": sites[nearest].name, "state": sites[nearest].state,
                    "distance_km": round(float(dist[nearest]), 1),
                    "direction": compass(from_here[nearest])},
        "weather_source": source, "note": note, "in_study_area": in_india,
        **here,
        "local": round(float(local[i_today]), 4),
        "incoming": round(float(inoculum_term(imported[i_today])), 4),
        "sources": sources,
    }

