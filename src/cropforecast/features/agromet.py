"""Block 2b - Climate feature engineering.

Turns the raw Open-Meteo daily archive into the "Agricultural Feature Set" of
the architecture diagram, following the diagram's own order:

    Raw Climate Data -> Feature Scaling -> Temporal Aggregation -> Feature Set

The derived quantities are the standard ones plant pathologists actually use -
vapour pressure deficit, dew-point depression, a leaf-wetness-duration proxy and
growing degree days - because they, not raw temperature, are what drive
infection models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Base temperature for growing degree days (deg C), typical for the crops here.
GDD_BASE = 10.0


def saturation_vapour_pressure(temp_c: np.ndarray | pd.Series) -> np.ndarray:
    """Saturation vapour pressure in kPa (Tetens equation)."""
    t = np.asarray(temp_c, dtype=float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def derive_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically derived daily variables to the raw archive.

    Expects the Open-Meteo daily columns; returns a copy with extra columns.
    """
    out = df.copy()

    t_mean = out["temperature_2m_mean"].astype(float)
    t_max = out["temperature_2m_max"].astype(float)
    t_min = out["temperature_2m_min"].astype(float)
    rh_mean = out["relative_humidity_2m_mean"].astype(float)
    rh_max = out["relative_humidity_2m_max"].astype(float)
    rain = out["precipitation_sum"].astype(float).fillna(0.0)

    # --- Vapour pressure deficit: the drying power of the air ---------------
    es = saturation_vapour_pressure(t_mean)
    ea = es * (rh_mean / 100.0)
    out["vpd_kpa"] = np.clip(es - ea, 0.0, None)

    # --- Dew point depression: small values mean condensation is imminent ---
    if "dew_point_2m_mean" in out.columns:
        out["dewpoint_depression_c"] = t_mean - out["dew_point_2m_mean"].astype(float)
    else:
        out["dewpoint_depression_c"] = np.nan

    # --- Diurnal temperature range: proxy for radiative cooling / dew -------
    out["diurnal_range_c"] = t_max - t_min

    # --- Leaf wetness duration proxy (hours) -------------------------------
    # Direct LWD sensors are rare, so the accepted practice is an RH-threshold
    # proxy: hours near saturation, plus the wetting contributed by rainfall.
    rh_component = np.clip((rh_max - 80.0) / 20.0, 0.0, 1.0) * 12.0
    rain_component = np.clip(rain / 10.0, 0.0, 1.0) * 8.0
    out["leaf_wetness_hours"] = np.clip(rh_component + rain_component, 0.0, 24.0)

    # --- Growing degree days ------------------------------------------------
    out["gdd"] = np.clip((t_max + t_min) / 2.0 - GDD_BASE, 0.0, None)

    # --- Wind decomposition: spore transport is directional ----------------
    if "wind_direction_10m_dominant" in out.columns:
        theta = np.deg2rad(out["wind_direction_10m_dominant"].astype(float))
        speed = out["wind_speed_10m_max"].astype(float)
        # Meteorological convention: direction is where wind blows FROM.
        # The transport vector therefore points the opposite way.
        out["wind_u"] = -speed * np.sin(theta)   # eastward component
        out["wind_v"] = -speed * np.cos(theta)   # northward component

    out["is_rainy"] = (rain >= 1.0).astype(float)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 7, 14),
    group_col: str = "site_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Temporal aggregation: trailing means/sums over several windows.

    Disease pressure is cumulative - a single humid day rarely matters, but a
    fortnight of them does. These trailing windows are what give the model its
    memory, and they are computed per site so no history crosses between farms.
    """
    out = df.sort_values([group_col, date_col]).copy()

    mean_vars = [
        "temperature_2m_mean", "relative_humidity_2m_mean", "vpd_kpa",
        "leaf_wetness_hours", "dewpoint_depression_c", "wind_speed_10m_max",
        "shortwave_radiation_sum",
    ]
    sum_vars = ["precipitation_sum", "gdd", "is_rainy"]

    grouped = out.groupby(group_col, sort=False)
    new_cols: dict[str, pd.Series] = {}
    for w in windows:
        for col in mean_vars:
            if col in out.columns:
                new_cols[f"{col}_mean{w}"] = grouped[col].transform(
                    lambda s, w=w: s.rolling(w, min_periods=1).mean()
                )
        for col in sum_vars:
            if col in out.columns:
                new_cols[f"{col}_sum{w}"] = grouped[col].transform(
                    lambda s, w=w: s.rolling(w, min_periods=1).sum()
                )

    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def build_climate_features(
    raw: pd.DataFrame, windows: tuple[int, ...] = (3, 7, 14)
) -> pd.DataFrame:
    """Full Block-2b pipeline: raw archive -> engineered agricultural features."""
    return add_rolling_features(derive_daily_features(raw), windows=windows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The numeric climate columns a model should consume.

    Identifiers and coordinates are excluded - location enters the model through
    the graph structure, not as a raw feature the network could memorise.
    """
    drop = {"site_id", "date", "lat", "lon", "wind_direction_10m_dominant"}
    return [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
