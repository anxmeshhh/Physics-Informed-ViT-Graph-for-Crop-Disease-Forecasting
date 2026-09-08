"""The agronomic knowledge base: how weather drives each PlantVillage disease.

This module is the "physics" in physics-informed learning. It encodes, for every
one of the 38 PlantVillage classes, the published environmental response of the
pathogen (or of the insect vector, where the disease is vector-borne).

It is used in two deliberately *different* ways, and keeping them different is
what stops the whole study being circular:

1. **Quantitatively**, to build the benchmark. Each image is placed at a real
   (farm, date) whose *real measured weather* was plausible for that disease.
   This is what gives the climate branch genuine signal to learn.

2. **Qualitatively**, in the physics-informed loss (:mod:`..physics.losses`).
   The loss never sees the functions below. It only enforces weak, directional
   statements - "risk must not fall when leaf wetness rises", "risk must be
   smooth across neighbouring farms" - so the network has to *learn* the
   quantitative relationship from data rather than being handed it.

Temperature response uses the Analytis / Yan-Hunt beta model, the standard
non-linear form in plant epidemiology: it is zero outside the cardinal
temperatures and peaks at exactly 1.0 at the optimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

MoistureMode = Literal["wet", "humid_no_free_water", "splash", "dry_hot",
                       "vector_warm", "neutral"]


@dataclass(frozen=True)
class PathogenProfile:
    """Environmental response of one pathogen / vector."""

    t_min: float          # no development below this (deg C)
    t_opt: float          # optimum
    t_max: float          # no development above this
    moisture: MoistureMode
    wetness_hours: float = 10.0   # leaf wetness needed for infection ("wet" mode)
    note: str = ""


# --- Per-class profiles ----------------------------------------------------
# Cardinal temperatures follow the standard plant-pathology literature for each
# organism. Healthy classes are handled separately (see `favourability`).
PROFILES: dict[str, PathogenProfile] = {
    # ---- Apple ----
    "Apple___Apple_scab": PathogenProfile(
        1, 18, 30, "wet", 12, "Venturia inaequalis; classic Mills leaf-wetness periods"),
    "Apple___Black_rot": PathogenProfile(
        7, 26, 32, "wet", 9, "Botryosphaeria obtusa; warm and wet"),
    "Apple___Cedar_apple_rust": PathogenProfile(
        6, 18, 27, "wet", 8, "Gymnosporangium juniperi-virginianae; needs prolonged wetness"),

    # ---- Cherry ----
    "Cherry_(including_sour)___Powdery_mildew": PathogenProfile(
        10, 23, 32, "humid_no_free_water", 0, "Podosphaera clandestina; free water suppresses"),

    # ---- Maize ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": PathogenProfile(
        14, 27, 35, "wet", 12, "Cercospora zeae-maydis; warm, extended humidity"),
    "Corn_(maize)___Common_rust_": PathogenProfile(
        8, 20, 30, "wet", 6, "Puccinia sorghi; cool-moderate with dew"),
    "Corn_(maize)___Northern_Leaf_Blight": PathogenProfile(
        10, 22, 30, "wet", 10, "Exserohilum turcicum; moderate temperature, long dew"),

    # ---- Grape ----
    "Grape___Black_rot": PathogenProfile(
        9, 26, 33, "wet", 8, "Guignardia bidwellii; warm rain events"),
    "Grape___Esca_(Black_Measles)": PathogenProfile(
        12, 30, 40, "dry_hot", 0, "Esca complex; expressed under heat and water stress"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": PathogenProfile(
        12, 25, 33, "wet", 10, "Pseudocercospora vitis; warm humid monsoon"),

    # ---- Citrus ----
    "Orange___Haunglongbing_(Citrus_greening)": PathogenProfile(
        16, 27, 35, "vector_warm", 0, "Vector Diaphorina citri; warm, flush-driven"),

    # ---- Peach ----
    "Peach___Bacterial_spot": PathogenProfile(
        13, 26, 35, "splash", 0, "Xanthomonas arboricola; rain-splash and wind-driven"),

    # ---- Pepper ----
    "Pepper,_bell___Bacterial_spot": PathogenProfile(
        15, 28, 36, "splash", 0, "Xanthomonas euvesicatoria; warm, rain-splash"),

    # ---- Potato ----
    "Potato___Early_blight": PathogenProfile(
        12, 27, 34, "wet", 8, "Alternaria solani; warm, alternating wet and dry"),
    "Potato___Late_blight": PathogenProfile(
        4, 17, 26, "wet", 11, "Phytophthora infestans; cool and wet - Smith periods"),

    # ---- Squash ----
    "Squash___Powdery_mildew": PathogenProfile(
        10, 27, 34, "humid_no_free_water", 0, "Podosphaera xanthii; dry leaves, humid air"),

    # ---- Strawberry ----
    "Strawberry___Leaf_scorch": PathogenProfile(
        10, 23, 30, "wet", 10, "Diplocarpon earlianum; wet foliage"),

    # ---- Tomato ----
    "Tomato___Bacterial_spot": PathogenProfile(
        15, 28, 36, "splash", 0, "Xanthomonas spp.; warm, rain-splash"),
    "Tomato___Early_blight": PathogenProfile(
        12, 28, 34, "wet", 8, "Alternaria solani"),
    "Tomato___Late_blight": PathogenProfile(
        4, 17, 26, "wet", 11, "Phytophthora infestans; cool wet nights"),
    "Tomato___Leaf_Mold": PathogenProfile(
        10, 23, 30, "humid_no_free_water", 0,
        "Passalora fulva; needs RH above ~85 percent, typical of enclosed humid canopies"),
    "Tomato___Septoria_leaf_spot": PathogenProfile(
        13, 23, 30, "wet", 10, "Septoria lycopersici; splash plus wetness"),
    "Tomato___Spider_mites Two-spotted_spider_mite": PathogenProfile(
        16, 31, 40, "dry_hot", 0,
        "Tetranychus urticae; outbreaks under hot dry stress - the inverse of fungal risk"),
    "Tomato___Target_Spot": PathogenProfile(
        15, 27, 34, "wet", 10, "Corynespora cassiicola"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": PathogenProfile(
        17, 30, 38, "dry_hot", 0, "Vector Bemisia tabaci; hot dry weather favours whitefly"),
    "Tomato___Tomato_mosaic_virus": PathogenProfile(
        10, 24, 34, "neutral", 0, "ToMV; mechanically transmitted, weak weather dependence"),
}


def temperature_response(
    t: np.ndarray | pd.Series, p: PathogenProfile
) -> np.ndarray:
    """Analytis / Yan-Hunt beta model, normalised to peak at 1.0 at ``t_opt``."""
    t = np.asarray(t, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        exponent = (p.t_opt - p.t_min) / (p.t_max - p.t_opt)
        left = (t - p.t_min) / (p.t_opt - p.t_min)
        right = (p.t_max - t) / (p.t_max - p.t_opt)
        resp = np.power(np.clip(left, 0, None), exponent) * np.clip(right, 0, None)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
    # Outside the cardinal range the organism does not develop at all.
    return np.clip(np.where((t <= p.t_min) | (t >= p.t_max), 0.0, resp), 0.0, 1.0)


def moisture_response(df: pd.DataFrame, p: PathogenProfile) -> np.ndarray:
    """Moisture term, whose *form depends on the pathogen's biology*."""
    rh = df["relative_humidity_2m_mean"].to_numpy(dtype=float)
    rain = df["precipitation_sum"].to_numpy(dtype=float)
    lwd = df["leaf_wetness_hours"].to_numpy(dtype=float)
    vpd = df["vpd_kpa"].to_numpy(dtype=float)
    wind = df["wind_speed_10m_max"].to_numpy(dtype=float)

    if p.moisture == "wet":
        # Infection needs a minimum accumulated wetness period.
        return np.clip(lwd / max(p.wetness_hours, 1e-6), 0.0, 1.0)

    if p.moisture == "humid_no_free_water":
        # Powdery mildews want humid air but free water bursts their conidia.
        humid = np.clip((rh - 55.0) / 35.0, 0.0, 1.0)
        washed_off = 1.0 - 0.75 * np.clip(rain / 12.0, 0.0, 1.0)
        return humid * washed_off

    if p.moisture == "splash":
        # Bacterial diseases spread by rain splash driven into wounds by wind.
        splash = np.clip(rain / 12.0, 0.0, 1.0)
        wind_term = np.clip(wind / 35.0, 0.0, 1.0)
        humid = np.clip((rh - 50.0) / 40.0, 0.0, 1.0)
        return np.clip(0.6 * splash + 0.2 * wind_term + 0.2 * humid, 0.0, 1.0)

    if p.moisture == "dry_hot":
        # Mites and whitefly thrive when the air is dry: high VPD, little rain.
        dryness = np.clip(vpd / 2.2, 0.0, 1.0)
        no_rain = 1.0 - np.clip(rain / 8.0, 0.0, 1.0)
        return np.clip(0.75 * dryness + 0.25 * no_rain, 0.0, 1.0)

    if p.moisture == "vector_warm":
        # Psyllid activity: suppressed by heavy rain, otherwise broadly tolerant.
        return np.clip(0.75 - 0.4 * np.clip(rain / 20.0, 0.0, 1.0)
                       + 0.25 * np.clip((rh - 40.0) / 40.0, 0.0, 1.0), 0.0, 1.0)

    return np.full(len(df), 0.5)   # "neutral"


def favourability(class_name: str, df: pd.DataFrame) -> np.ndarray:
    """Environmental favourability in [0, 1] for ``class_name`` on each row.

    For a diseased class this is the classical product of a temperature term and
    a moisture term. For a *healthy* class it is the complement of the worst
    disease pressure facing that same crop - a healthy leaf is most plausible
    exactly where conditions were hostile to its crop's pathogens.
    """
    if class_name in PROFILES:
        p = PROFILES[class_name]
        temp = temperature_response(df["temperature_2m_mean"], p)
        moist = moisture_response(df, p)
        return np.clip(temp * moist, 0.0, 1.0)

    if class_name.endswith("___healthy"):
        crop = class_name.split("___", 1)[0]
        peers = [c for c in PROFILES if c.split("___", 1)[0] == crop]
        if not peers:
            # Crops PlantVillage only photographs healthy (blueberry, raspberry,
            # soybean): use a mild generic fungal pressure as the foil.
            generic = PathogenProfile(8, 24, 34, "wet", 10)
            pressure = temperature_response(df["temperature_2m_mean"], generic) \
                * moisture_response(df, generic)
        else:
            pressure = np.max(
                np.stack([favourability(c, df) for c in peers]), axis=0
            )
        return np.clip(1.0 - pressure, 0.0, 1.0)

    raise KeyError(f"No pathogen profile for class {class_name!r}")


def crop_pressure(crop: str, df: pd.DataFrame) -> np.ndarray:
    """Worst-case disease pressure across every pathogen of one crop.

    This is the aggregate agronomic risk a farmer actually cares about, and it
    is what the forecasting head is trained to anticipate.
    """
    peers = [c for c in PROFILES if c.split("___", 1)[0] == crop]
    if not peers:
        generic = PathogenProfile(8, 24, 34, "wet", 10)
        return np.clip(
            temperature_response(df["temperature_2m_mean"], generic)
            * moisture_response(df, generic), 0.0, 1.0
        )
    return np.max(np.stack([favourability(c, df) for c in peers]), axis=0)


def disease_classes() -> list[str]:
    """Every class with an explicit pathogen profile."""
    return sorted(PROFILES)
