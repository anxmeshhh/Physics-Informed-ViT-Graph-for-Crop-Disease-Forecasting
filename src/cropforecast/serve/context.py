"""Why a diagnosis came out the way it did: leaf -> disease -> weather -> risk.

A diagnosis on its own leaves the obvious questions open. Does the weather at
this farm actually suit *this* disease? Why does the district map say "act now"
when the forecast for this farm is low? This module answers both from the same
physics everything else uses, so every number on the result can be traced:

* **This disease vs this weather** - the pathogen's cardinal temperatures and
  moisture need (the knowledge base in ``physics.epidemiology``) against the
  weather recorded at the farm on that date, and how favourable that makes it
  over the coming week.
* **The farm's crop as a whole** - exactly the district card on the bulletin:
  the weather's worst-case pressure across every disease of this crop.
* **Is the disease here, or arriving?** - the simulated epidemic at this farm,
  and what the neighbouring districts are sending on the wind.

The model's risk forecast was trained on conduciveness *and* inoculum together
(``data.assign.attach_forecast_targets``), which is why favourable weather with
no disease anywhere near still forecasts low - and the result now says so.
"""
from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from ..physics.contagion import inoculum_term
from ..physics.epidemiology import (PROFILES, crop_pressure, moisture_response,
                                    temperature_response)
from . import bulletin, nearby

MOISTURE_NEED = {
    "wet": "{h:g}+ hours of wet leaves",
    "humid_no_free_water": "humid air, but rain washes it off",
    "splash": "rain splash and wind",
    "dry_hot": "hot, dry air and little rain",
    "vector_warm": "warm weather for its insect carrier; heavy rain holds it back",
    "neutral": "little - it spreads by contact",
}


def _grade(score: float) -> str:
    return "favourable" if score >= 0.6 else "partly favourable" if score >= 0.25 else "unfavourable"


def disease_vs_weather(disease: str, weather: pd.DataFrame, today: pd.Timestamp) -> dict | None:
    """The diagnosed disease's needs against the weather recorded for it."""
    if disease not in PROFILES:
        return None
    p = PROFILES[disease]
    ahead = weather[weather.date >= today].reset_index(drop=True)
    temp = temperature_response(ahead.temperature_2m_mean, p)
    moist = moisture_response(ahead, p)
    fav = np.clip(temp * moist, 0.0, 1.0)
    now = ahead.iloc[0]
    t = float(now.temperature_2m_mean)
    if temp[0] >= 0.6:
        t_note = "close to its ideal"
    else:
        t_note = "too warm" if t > p.t_opt else "too cool"

    if p.moisture == "wet":
        m_obs = f"{now.leaf_wetness_hours:.1f} h of wet leaves"
    elif p.moisture == "humid_no_free_water":
        m_obs = f"{now.relative_humidity_2m_mean:.0f}% humidity, {now.precipitation_sum:.1f} mm rain"
    elif p.moisture == "splash":
        m_obs = f"{now.precipitation_sum:.1f} mm rain, wind {now.wind_speed_10m_max:.0f} km/h"
    elif p.moisture == "dry_hot":
        m_obs = f"VPD {now.vpd_kpa:.2f} kPa, {now.precipitation_sum:.1f} mm rain"
    else:
        m_obs = f"{now.relative_humidity_2m_mean:.0f}% humidity, {now.precipitation_sum:.1f} mm rain"

    k = int(np.argmax(fav))
    return {
        "disease": disease, "name": bulletin.pretty(disease),
        "needs": {
            "temperature": f"{p.t_min:g}-{p.t_max:g} °C, best near {p.t_opt:g} °C",
            "moisture": MOISTURE_NEED[p.moisture].format(h=p.wetness_hours),
        },
        "observed": {"temperature": f"{t:.1f} °C", "moisture": m_obs},
        "temperature_score": round(float(temp[0]), 3), "temperature_note": t_note,
        "temperature_grade": _grade(float(temp[0])),
        "moisture_score": round(float(moist[0]), 3), "moisture_grade": _grade(float(moist[0])),
        "today": round(float(fav[0]), 3), "series": [round(float(v), 3) for v in fav],
        "peak": round(float(fav[k]), 3), "peak_day": k,
        "band": bulletin.band(float(fav.max())),
    }


def case_context(climate: pd.DataFrame, outbreaks: pd.DataFrame | None, crop: str,
                 disease: str, site, when: Date) -> dict:
    start = when - timedelta(days=bulletin.TREND_DAYS - 1)
    end = when + timedelta(days=bulletin.OUTLOOK_DAYS)
    days = pd.date_range(start, end)
    today = pd.Timestamp(when)
    w = (climate[(climate.site_id == site.site_id) & climate.date.isin(days)]
         .sort_values("date").reset_index(drop=True))
    peers = [c for c in PROFILES if c.split("___", 1)[0] == crop]

    # Exactly the bulletin's card for this district.
    district = bulletin.assess(crop_pressure(crop, w), w, today, peers)

    # Present at this farm (the simulated epidemic) and arriving from the rest.
    here = 0.0
    if outbreaks is not None:
        row = outbreaks[(outbreaks.crop == crop) & (outbreaks.site_id == site.site_id)
                        & (outbreaks.date == today)]
        if len(row):
            here = float(inoculum_term(row.infection_pressure.iloc[0]))
    arr = nearby.arriving(climate, outbreaks, crop, days, site.lat, site.lon, exclude=site.site_id)
    i_today = list(days).index(today)
    incoming = float(inoculum_term(arr["imported"][i_today]))
    top = None
    if arr["contrib"][i_today].sum() > 0:
        j = int(np.argmax(arr["contrib"][i_today]))
        s = arr["sites"][j]
        top = {"name": s.name, "distance_km": round(float(arr["dist"][j]), 1),
               "share": round(float(arr["contrib"][i_today, j] / arr["contrib"][i_today].sum()), 3)}

    return {
        "crop_label": bulletin.crop_label(crop),
        "this_disease": disease_vs_weather(disease, w, today),
        "district": {
            "alert": district["alert"], "alert_text": district["alert_text"],
            "peak_risk": district["peak_risk"], "peak_day": district["peak_day"],
            "threat": district["threat"]["name"] if district["threat"] else None,
        },
        "disease_present": round(here, 3),
        "disease_arriving": round(incoming, 3),
        "arriving_from": top,
    }
