"""District alert bulletin: the forecast restated as decisions an officer can act on.

The map answers "how risky is each district today?". An agriculture officer needs
more than that: which districts need action this week, what the threat is, which
way it is moving, and what to do about it. Every number here comes from the same
physics the network is trained against - pathogen favourability computed from the
weather actually recorded at each district - so the bulletin and the model never
disagree about what "risk" means.

The 7-day outlook is a hindcast. The archive is reanalysis, so the weather of the
following week is known and stands in for a weather forecast; in deployment the
same computation would run on forecast weather instead (see the project roadmap).
The page says so wherever the outlook is shown.

Recommended actions are deliberately generic - the kind of measure an extension
service advises for a class of pathogen - and defer product and dose to the state
agriculture department. The system ranks and explains risk; it does not prescribe.
"""
from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from ..data.farms import SITES, to_frame
from ..physics.epidemiology import PROFILES, crop_pressure, favourability

RISK_LABELS = ("Low", "Medium", "High")
OUTLOOK_DAYS = 7
TREND_DAYS = 7


def band(pressure: float) -> str:
    """Same cut-points as the regional map, so a district never changes colour
    between the map and the bulletin."""
    return "High" if pressure > 0.55 else "Medium" if pressure > 0.25 else "Low"


# --- what kind of organism, and what is done about it ------------------------
ORGANISM = {
    "Apple___Apple_scab": "fungus", "Apple___Black_rot": "fungus",
    "Apple___Cedar_apple_rust": "fungus",
    "Cherry_(including_sour)___Powdery_mildew": "fungus",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "fungus",
    "Corn_(maize)___Common_rust_": "fungus",
    "Corn_(maize)___Northern_Leaf_Blight": "fungus",
    "Grape___Black_rot": "fungus", "Grape___Esca_(Black_Measles)": "fungus",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "fungus",
    "Potato___Early_blight": "fungus", "Squash___Powdery_mildew": "fungus",
    "Strawberry___Leaf_scorch": "fungus", "Tomato___Early_blight": "fungus",
    "Tomato___Leaf_Mold": "fungus", "Tomato___Septoria_leaf_spot": "fungus",
    "Tomato___Target_Spot": "fungus",
    "Potato___Late_blight": "water_mould", "Tomato___Late_blight": "water_mould",
    "Peach___Bacterial_spot": "bacterium", "Pepper,_bell___Bacterial_spot": "bacterium",
    "Tomato___Bacterial_spot": "bacterium",
    "Orange___Haunglongbing_(Citrus_greening)": "insect_borne",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "insect_borne",
    "Tomato___Tomato_mosaic_virus": "contact_virus",
    "Tomato___Spider_mites Two-spotted_spider_mite": "mite",
}

ORGANISM_LABEL = {
    "fungus": "Fungal disease",
    "water_mould": "Water-mould disease - spreads fast",
    "bacterium": "Bacterial disease",
    "insect_borne": "Spread by insects",
    "contact_virus": "Virus spread by contact",
    "mite": "Mite pest",
}

FARMER_ACTIONS = {
    "fungus": [
        "Preventive fungicide spray, as recommended by the state agriculture department",
        "Remove and destroy infected leaves",
        "Avoid overhead irrigation; keep the canopy open for airflow",
    ],
    "water_mould": [
        "Spray before the next wet spell - this disease can destroy a field within days",
        "Uproot infected plants and keep them out of compost and cull piles",
        "Avoid evening irrigation, which keeps leaves wet overnight",
    ],
    "bacterium": [
        "Copper-based protective spray, as recommended by the state agriculture department",
        "Do not work in the field while plants are wet",
        "Use certified disease-free seed and transplants",
    ],
    "insect_borne": [
        "Control the insect carrier (whitefly / psyllid); yellow sticky traps for monitoring",
        "Uproot and destroy infected plants - there is no cure once infected",
        "Prefer tolerant varieties for the next sowing",
    ],
    "contact_virus": [
        "Remove infected plants; disinfect tools and wash hands between plants",
        "No chemical cure - prevention is the only control",
    ],
    "mite": [
        "Scout the undersides of leaves twice a week",
        "Water sprays and adequate irrigation slow mite build-up",
        "Use a miticide only once the local threshold is crossed",
    ],
}

OFFICER_ACTIONS = {
    "High": [
        "Send an SMS advisory to farmers in this district today",
        "Ask the district agriculture officer / KVK to verify in the field within 48 hours",
        "Check stock of recommended plant-protection inputs at local depots",
    ],
    "Medium": [
        "Advise farmers to scout their fields twice a week",
        "Keep extension staff ready to respond if the alert rises",
    ],
    "Low": ["Routine monitoring - no action needed this week"],
}

ALERT_TEXT = {"High": "Act now", "Medium": "Watch", "Low": "Normal"}


def case_officer_actions(disease: str, alert: str) -> list[str]:
    """Officer actions for one diagnosed leaf.

    A confirmed infection is never "no action", even when the weather will not
    spread it this week - it still has to be recorded and rechecked.
    """
    if disease.endswith("___healthy") or alert != "Low":
        return OFFICER_ACTIONS[alert]
    return [
        "Record the confirmed case for this farm in the district register",
        "Weather does not favour spread this week - recheck if it turns wet or humid",
    ]


# PlantVillage folder names, restated the way an officer would say them.
DISPLAY = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Grey leaf spot (Cercospora)",
    "Corn_(maize)___Common_rust_": "Common rust",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider mites",
    "Grape___Esca_(Black_Measles)": "Esca (black measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight (Isariopsis)",
    "Orange___Haunglongbing_(Citrus_greening)": "Citrus greening (HLB)",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus",
}
CROP_DISPLAY = {"Corn_(maize)": "Maize", "Pepper,_bell": "Bell pepper",
                "Cherry_(including_sour)": "Cherry"}


def pretty(class_name: str) -> str:
    if class_name in DISPLAY:
        return DISPLAY[class_name]
    name = class_name.split("___", 1)[-1].replace("_", " ").strip()
    return name[:1].upper() + name[1:]


def crop_label(crop: str) -> str:
    return CROP_DISPLAY.get(crop, crop.replace("_", " "))


def farmer_actions(disease: str) -> list[str]:
    """Measures for the farmer; empty for a healthy leaf."""
    return FARMER_ACTIONS.get(ORGANISM.get(disease, ""), [])


def threat_profile(disease: str) -> dict:
    org = ORGANISM.get(disease)
    return {"class": disease, "name": pretty(disease),
            "organism": ORGANISM_LABEL.get(org, "Disease pressure"),
            "farmer_actions": FARMER_ACTIONS.get(org, FARMER_ACTIONS["fungus"])}


# --- the graph a district's neighbours come from ------------------------------
def crop_edges(cfg, crop: str) -> tuple[pd.DataFrame, list[dict]]:
    """Farms growing ``crop`` and the spatial-graph edges among them.

    Same rule as graph construction: KNN within the search radius, topped up to
    the isolation floor so no district is left without neighbours.
    """
    from ..graph.build import site_distance_matrix

    sites = [s for s in SITES if crop in s.crops]
    if len(sites) < 2:
        return to_frame(tuple(sites)), []

    sub = to_frame(tuple(sites))
    dist, bearing, ids = site_distance_matrix(sub)
    radius, k = cfg.graph.radius_km, cfg.graph.k_neighbours
    min_nb = cfg.graph.min_neighbours

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
    return sub, edges


# --- the bulletin -------------------------------------------------------------
def _drivers(row: pd.Series, disease: str | None, alert: str) -> str:
    """The weather that explains the alert, in the terms that matter for it."""
    t = row.temperature_2m_mean
    rh = row.relative_humidity_2m_mean
    wet = row.leaf_wetness_hours
    rain = row.precipitation_sum
    if alert == "Low":
        return f"Unfavourable for disease: {rh:.0f}% humidity, {wet:.0f} h leaf wetness, {t:.0f} °C"
    mode = PROFILES[disease].moisture if disease in PROFILES else "wet"
    if mode == "dry_hot":
        return f"Hot and dry: {t:.0f} °C, {rh:.0f}% humidity"
    if mode == "humid_no_free_water":
        return f"Humid air: {rh:.0f}% humidity at {t:.0f} °C"
    if mode == "splash":
        return f"Rain splash: {rain:.0f} mm rain, {t:.0f} °C"
    return f"Wet leaves: {wet:.0f} h leaf wetness, {rh:.0f}% humidity, {t:.0f} °C"


def build(cfg, climate: pd.DataFrame, crop: str, when: Date) -> dict:
    """The bulletin for every district growing ``crop``, as of ``when``."""
    start = pd.Timestamp(when - timedelta(days=TREND_DAYS - 1))
    end = pd.Timestamp(when + timedelta(days=OUTLOOK_DAYS))
    today = pd.Timestamp(when)

    sub_sites, edges = crop_edges(cfg, crop)
    peers = [c for c in PROFILES if c.split("___", 1)[0] == crop]

    window = climate[(climate.date >= start) & (climate.date <= end)
                     & climate.site_id.isin(sub_sites.site_id)].sort_values("date")

    rows = []
    for s in sub_sites.itertuples():
        w = window[window.site_id == s.site_id].reset_index(drop=True)
        if not (w.date == today).any():
            continue
        pressure = crop_pressure(crop, w)
        dates = list(w.date)
        i_today = dates.index(today)

        past = pressure[: i_today + 1]
        ahead = pressure[i_today + 1:]
        risk = float(pressure[i_today])

        # The worst day in view, today included: that is what the officer has
        # to prepare for, and naming a lower "peak" than today would read wrong.
        k = int(np.argmax(pressure[i_today:]))
        at = i_today + k
        peak, peak_day = float(pressure[at]), (dates[at] - today).days
        alert = band(peak)

        # A normal district has no threat worth naming.
        threat = None
        if peers and alert != "Low":
            scores = {c: float(favourability(c, w.iloc[[at]])[0]) for c in peers}
            threat = max(scores, key=scores.get)

        change = risk - float(past[0])
        rows.append({
            "site_id": s.site_id, "name": s.name, "state": s.state,
            "lat": s.lat, "lon": s.lon,
            "risk": round(risk, 4), "band": band(risk),
            "peak_risk": round(peak, 4), "peak_band": band(peak), "peak_day": peak_day,
            "alert": alert, "alert_text": ALERT_TEXT[alert],
            "trend": [round(float(v), 4) for v in past],
            "outlook": [round(float(v), 4) for v in ahead],
            "change_7d": round(change, 4),
            "direction": ("Rising" if change > 0.08 else
                          "Falling" if change < -0.08 else "Steady"),
            "threat": threat_profile(threat) if threat else None,
            "drivers": _drivers(w.iloc[at], threat, alert),
            "officer_actions": OFFICER_ACTIONS[alert],
        })

    # Spread watch: the graph the model passes messages over, read as "which
    # neighbouring districts are already on alert".
    alert_of = {r["site_id"]: r for r in rows}
    for r in rows:
        nbrs = {e["target"]: e["distance_km"] for e in edges if e["source"] == r["site_id"]}
        r["neighbours_high"] = [
            {"name": alert_of[n]["name"], "distance_km": d}
            for n, d in sorted(nbrs.items(), key=lambda kv: kv[1])
            if n in alert_of and alert_of[n]["alert"] == "High"
        ]

    order = {"High": 0, "Medium": 1, "Low": 2}
    rows.sort(key=lambda r: (order[r["alert"]], -max(r["risk"], r["peak_risk"])))

    counts = {b: sum(r["alert"] == b for r in rows) for b in RISK_LABELS}
    act = [r for r in rows if r["alert"] == "High"]
    threats = pd.Series([r["threat"]["name"] for r in act if r["threat"]])
    main = threats.value_counts().index[0] if len(threats) else None
    rising = sum(r["direction"] == "Rising" for r in rows)

    crop_name = crop_label(crop)
    if act:
        headline = (f"{len(act)} of {len(rows)} {crop_name.lower()} districts need action "
                    f"this week" + (f" - main threat: {main}" if main else ""))
    elif counts["Medium"]:
        headline = (f"No district needs urgent action; {counts['Medium']} of {len(rows)} "
                    f"{crop_name.lower()} districts are on watch")
    else:
        headline = f"All {len(rows)} {crop_name.lower()} districts are at normal risk this week"

    return {
        "crop": crop, "crop_label": crop_name, "date": str(when),
        "outlook_until": str(when + timedelta(days=OUTLOOK_DAYS)),
        "headline": headline,
        "summary": {
            "districts": len(rows), "counts": counts, "rising": rising,
            "states_on_alert": sorted({r["state"] for r in act}),
            "main_threat": main,
        },
        "districts": rows,
        "edges": edges,
    }
