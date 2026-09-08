"""Block 2b - Climate data acquisition (Open-Meteo historical archive).

Pulls *real* daily meteorology for every farm site in the registry. Open-Meteo's
archive endpoint is ERA5-derived reanalysis, is free, needs no API key, and
returns the true surface elevation of the queried grid cell - which we keep as
the site elevation rather than hard-coding one.

Everything is cached to ``data/climate/<site_id>.csv``. After the first run the
whole pipeline is fully offline and reproducible.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .farms import SITES, FarmSite

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables requested from the archive. These map onto the "Climate Data" box of
# the architecture diagram: temperature, humidity, rainfall, wind, solar, pressure.
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_max",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "surface_pressure_mean",
    "dew_point_2m_mean",
]


def fetch_site(
    site: FarmSite,
    start_date: str,
    end_date: str,
    daily_vars: list[str] | None = None,
    url: str = ARCHIVE_URL,
    timeout: int = 90,
    max_retries: int = 4,
) -> pd.DataFrame:
    """Fetch the full daily archive for one site as a tidy DataFrame."""
    daily_vars = daily_vars or DAILY_VARS
    params = {
        "latitude": site.lat,
        "longitude": site.lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": "auto",
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                # Free tier throttles hard. Honour Retry-After when present,
                # otherwise back off generously - a slow fetch is fine because
                # the result is cached forever.
                wait = float(resp.headers.get("Retry-After", 0)) or 20.0 * (attempt + 1)
                print(f"      rate-limited, waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:      # network hiccup
            last_err = exc
            time.sleep(5.0 * (attempt + 1))
    else:
        raise RuntimeError(f"Open-Meteo failed for {site.site_id}: {last_err}")

    daily = payload["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "site_id", site.site_id)
    # The API reports the true elevation of the grid cell it served.
    df["elevation_m"] = payload.get("elevation")
    df["lat"] = payload.get("latitude", site.lat)
    df["lon"] = payload.get("longitude", site.lon)
    return df


def fetch_all(
    start_date: str,
    end_date: str,
    cache_dir: str | Path,
    sites: tuple[FarmSite, ...] = SITES,
    daily_vars: list[str] | None = None,
    force: bool = False,
    polite_delay: float = 0.6,
    max_retries: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch (or load from cache) the archive for every site.

    One HTTP call covers a site's entire multi-year range, so the whole registry
    costs ~37 requests.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for i, site in enumerate(sites, 1):
        cache_file = cache_dir / f"{site.site_id}.csv"
        if cache_file.exists() and not force:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            status = "cached"
        else:
            df = fetch_site(site, start_date, end_date, daily_vars,
                            max_retries=max_retries)
            df.to_csv(cache_file, index=False)
            status = "fetched"
            time.sleep(polite_delay)                  # be a good API citizen
        if verbose:
            print(f"  [{i:2d}/{len(sites)}] {site.site_id:8s} {site.name:24s} "
                  f"{status:7s} rows={len(df):5d} elev={df.elevation_m.iloc[0]:.0f}m")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["site_id", "date"]).reset_index(drop=True)


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-site data coverage and missing-value counts, for a sanity check."""
    value_cols = [c for c in df.columns
                  if c not in {"site_id", "date", "lat", "lon", "elevation_m"}]
    rows = []
    for site_id, g in df.groupby("site_id"):
        rows.append(
            {
                "site_id": site_id,
                "start": g.date.min().date(),
                "end": g.date.max().date(),
                "days": len(g),
                "elevation_m": g.elevation_m.iloc[0],
                "missing_cells": int(g[value_cols].isna().sum().sum()),
                "mean_temp_C": round(float(g.temperature_2m_mean.mean()), 1),
                "annual_rain_mm": round(float(g.precipitation_sum.sum() / max(len(g) / 365.25, 1e-9)), 0),
            }
        )
    return pd.DataFrame(rows).sort_values("site_id").reset_index(drop=True)
