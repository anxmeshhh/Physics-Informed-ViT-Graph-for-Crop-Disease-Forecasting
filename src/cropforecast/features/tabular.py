"""Assembling the model-ready tensors from the observations table.

Produces four matrices:

``climate``
    Every engineered agro-meteorological variable, standardised.
``meta``
    Farm metadata from the diagram's "Farm Location & Metadata" box: latitude,
    longitude, altitude, crop type, soil type and season.
``physics``
    The *raw, unscaled* physical drivers the physics loss reads. These must keep
    their real units - a constraint about vapour pressure deficit in kPa is
    meaningless applied to a z-score.
``targets``
    Disease label, plus continuous risk and risk band at each horizon.

Scalers and encoders are fitted on the **training split only** and then applied
to validation and test, so no distributional information leaks backwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..physics.losses import PHYSICS_COLUMNS

# Columns that must never be treated as model inputs.
#   label / is_healthy / risk_*  - targets
#   lat / lon / elevation_m      - belong to the metadata stream, not climate
#   wind_direction_*             - already decomposed into wind_u / wind_v
#   _row                         - internal positional index into the embedding
#                                  cache; numeric, so it would silently become a
#                                  feature and leak row ordering
_EXCLUDE = {
    "label", "elevation_m", "lat", "lon", "wind_direction_10m_dominant",
    "is_healthy", "_row",
}

META_NUMERIC = ["lat", "lon", "elevation_m"]
META_CATEGORICAL = ["crop", "soil_type", "agro_zone"]


@dataclass
class FeatureSpace:
    """Fitted transformers plus the resolved column lists."""

    climate_cols: list[str]
    scaler_climate: StandardScaler
    scaler_meta: StandardScaler
    encoder_meta: OneHotEncoder
    meta_categories: list[str] = field(default_factory=list)

    @property
    def climate_dim(self) -> int:
        return len(self.climate_cols)

    @property
    def meta_dim(self) -> int:
        return len(META_NUMERIC) + 2 + len(self.meta_categories)  # +2 season sin/cos


def climate_columns(obs: pd.DataFrame) -> list[str]:
    """Numeric climate columns, excluding identifiers and anything target-derived."""
    cols = []
    for c in obs.columns:
        if c in _EXCLUDE or c.startswith("risk_"):
            continue
        if not pd.api.types.is_numeric_dtype(obs[c]):
            continue
        cols.append(c)
    return cols


def _season_features(dates: pd.Series) -> np.ndarray:
    """Cyclical day-of-year encoding, so 31 Dec sits next to 1 Jan."""
    doy = pd.to_datetime(dates).dt.dayofyear.to_numpy(dtype=float)
    angle = 2.0 * np.pi * doy / 365.25
    return np.stack([np.sin(angle), np.cos(angle)], axis=1)


def fit_feature_space(train: pd.DataFrame) -> FeatureSpace:
    """Fit every scaler and encoder on the training split alone."""
    cols = climate_columns(train)

    scaler_climate = StandardScaler().fit(train[cols].to_numpy(dtype=np.float64))
    scaler_meta = StandardScaler().fit(train[META_NUMERIC].to_numpy(dtype=np.float64))
    encoder_meta = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder_meta.fit(train[META_CATEGORICAL])

    categories = [
        f"{col}={val}"
        for col, vals in zip(META_CATEGORICAL, encoder_meta.categories_)
        for val in vals
    ]
    return FeatureSpace(cols, scaler_climate, scaler_meta, encoder_meta, categories)


def transform(
    obs: pd.DataFrame, space: FeatureSpace, horizons: list[int]
) -> dict[str, np.ndarray]:
    """Apply a fitted feature space to any split."""
    climate = space.scaler_climate.transform(
        obs[space.climate_cols].to_numpy(dtype=np.float64)
    ).astype(np.float32)

    meta = np.concatenate(
        [
            space.scaler_meta.transform(obs[META_NUMERIC].to_numpy(dtype=np.float64)),
            _season_features(obs.date),
            space.encoder_meta.transform(obs[META_CATEGORICAL]),
        ],
        axis=1,
    ).astype(np.float32)

    physics = obs[list(PHYSICS_COLUMNS)].to_numpy(dtype=np.float32)

    risk_reg = obs[[f"risk_t{h}" for h in horizons]].to_numpy(dtype=np.float32)
    risk_cls = obs[[f"risk_level_t{h}" for h in horizons]].to_numpy(dtype=np.int64)
    labels = obs["label"].to_numpy(dtype=np.int64)

    return {
        "climate": np.nan_to_num(climate),
        "meta": np.nan_to_num(meta),
        "physics": np.nan_to_num(physics),
        "risk_reg": risk_reg,
        "risk_cls": risk_cls,
        "labels": labels,
    }


def attach_site_metadata(obs: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Join soil type and agro-climatic zone onto the observations."""
    cols = ["site_id", "soil_type", "agro_zone", "name", "state"]
    return obs.merge(sites[cols], on="site_id", how="left", validate="m:1")
