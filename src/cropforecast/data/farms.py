"""Block 1 - Farm location & metadata.

PlantVillage ships leaf photographs with no geography and no timestamps, but the
architecture needs both: the graph is built over farm coordinates, and the
climate layer needs a date to query. This module supplies the missing geography
as a registry of *real* Indian agricultural districts, each one an actual centre
of production for the crops it is listed against.

Nothing here is invented: coordinates are the district centroids, and the
crop-to-district mapping follows established Indian production belts (Nashik
grapes, Nagpur citrus, Agra potato, Kolar tomato, Shimla apple, and so on).
Elevation is deliberately *not* hard-coded - it is filled in from the Open-Meteo
response, which returns the true elevation of each queried grid cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FarmSite:
    """One node of the spatial graph."""

    site_id: str
    name: str
    state: str
    lat: float
    lon: float
    agro_zone: str      # ICAR agro-climatic zone
    soil_type: str
    crops: tuple[str, ...]
    elevation_m: float | None = field(default=None)


# --- The registry ----------------------------------------------------------
# Crop names use the PlantVillage vocabulary verbatim so the join is exact.
SITES: tuple[FarmSite, ...] = (
    # --- Western Himalayan temperate belt: apple, cherry, peach, raspberry ---
    FarmSite("HP-SML", "Shimla", "Himachal Pradesh", 31.1048, 77.1734,
             "Western Himalayan", "Brown hill soil",
             ("Apple", "Cherry_(including_sour)", "Peach", "Raspberry")),
    FarmSite("HP-KUL", "Kullu", "Himachal Pradesh", 31.9576, 77.1093,
             "Western Himalayan", "Brown hill soil",
             ("Apple", "Cherry_(including_sour)", "Raspberry")),
    FarmSite("HP-KNR", "Reckong Peo (Kinnaur)", "Himachal Pradesh", 31.5386, 78.2732,
             "Western Himalayan", "Skeletal mountain soil",
             ("Apple", "Raspberry")),
    FarmSite("JK-SGR", "Srinagar", "Jammu & Kashmir", 34.0837, 74.7973,
             "Western Himalayan", "Karewa alluvium",
             ("Apple", "Cherry_(including_sour)", "Peach")),
    FarmSite("JK-BRM", "Baramulla", "Jammu & Kashmir", 34.2090, 74.3420,
             "Western Himalayan", "Karewa alluvium",
             ("Apple", "Cherry_(including_sour)")),
    FarmSite("JK-ANT", "Anantnag", "Jammu & Kashmir", 33.7311, 75.1487,
             "Western Himalayan", "Karewa alluvium",
             ("Apple", "Cherry_(including_sour)")),
    FarmSite("HP-SOL", "Solan", "Himachal Pradesh", 30.9045, 77.0967,
             "Western Himalayan", "Brown hill soil",
             ("Peach", "Pepper,_bell", "Strawberry", "Tomato")),
    FarmSite("UK-DDN", "Dehradun", "Uttarakhand", 30.3165, 78.0322,
             "Western Himalayan", "Doon gravel loam",
             ("Peach", "Strawberry", "Tomato")),
    FarmSite("UK-NTL", "Nainital", "Uttarakhand", 29.3803, 79.4636,
             "Western Himalayan", "Brown forest soil",
             ("Strawberry", "Raspberry", "Apple")),
    FarmSite("HP-MNL", "Manali", "Himachal Pradesh", 32.2396, 77.1887,
             "Western Himalayan", "Brown hill soil",
             ("Apple", "Raspberry", "Cherry_(including_sour)")),

    # --- Deccan plateau: grape, tomato, cucurbits ---
    FarmSite("MH-NSK", "Nashik", "Maharashtra", 19.9975, 73.7898,
             "Western Plateau & Hills", "Medium black soil",
             ("Grape", "Tomato", "Squash")),
    FarmSite("MH-SNG", "Sangli", "Maharashtra", 16.8524, 74.5815,
             "Western Plateau & Hills", "Deep black soil",
             ("Grape", "Tomato")),
    FarmSite("MH-SLP", "Solapur", "Maharashtra", 17.6599, 75.9064,
             "Western Plateau & Hills", "Deep black soil",
             ("Grape", "Squash")),
    FarmSite("KA-VJP", "Vijayapura (Bijapur)", "Karnataka", 16.8302, 75.7100,
             "Southern Plateau & Hills", "Deep black soil",
             ("Grape", "Corn_(maize)")),
    FarmSite("MH-MHB", "Mahabaleshwar", "Maharashtra", 17.9307, 73.6477,
             "Western Plateau & Hills", "Lateritic hill soil",
             ("Strawberry", "Raspberry")),

    # --- Vidarbha citrus belt ---
    FarmSite("MH-NGP", "Nagpur", "Maharashtra", 21.1458, 79.0882,
             "Western Plateau & Hills", "Medium black soil",
             ("Orange", "Soybean", "Corn_(maize)")),
    FarmSite("MH-AMR", "Amravati", "Maharashtra", 20.9320, 77.7523,
             "Western Plateau & Hills", "Medium black soil",
             ("Orange", "Soybean")),

    # --- Indo-Gangetic potato & vegetable plains ---
    FarmSite("UP-AGR", "Agra", "Uttar Pradesh", 27.1767, 78.0081,
             "Upper Gangetic Plains", "Alluvial soil",
             ("Potato", "Tomato", "Squash", "Pepper,_bell")),
    FarmSite("UP-FRK", "Farrukhabad", "Uttar Pradesh", 27.3929, 79.5800,
             "Upper Gangetic Plains", "Alluvial soil",
             ("Potato", "Squash")),
    FarmSite("PB-JLD", "Jalandhar", "Punjab", 31.3260, 75.5762,
             "Trans-Gangetic Plains", "Alluvial soil",
             ("Potato", "Corn_(maize)", "Tomato")),
    FarmSite("WB-HGL", "Hooghly", "West Bengal", 22.8994, 88.3903,
             "Lower Gangetic Plains", "Alluvial soil",
             ("Potato", "Squash", "Tomato")),
    FarmSite("UP-LKO", "Lucknow", "Uttar Pradesh", 26.8467, 80.9462,
             "Upper Gangetic Plains", "Alluvial soil",
             ("Squash", "Tomato", "Pepper,_bell")),
    FarmSite("UP-VNS", "Varanasi", "Uttar Pradesh", 25.3176, 82.9739,
             "Middle Gangetic Plains", "Alluvial soil",
             ("Squash", "Pepper,_bell", "Tomato")),
    FarmSite("BR-PTN", "Patna", "Bihar", 25.5941, 85.1376,
             "Middle Gangetic Plains", "Calcareous alluvial",
             ("Squash", "Potato", "Corn_(maize)")),

    # --- Malwa plateau soybean bowl ---
    FarmSite("MP-IND", "Indore", "Madhya Pradesh", 22.7196, 75.8577,
             "Central Plateau & Hills", "Medium black soil",
             ("Soybean", "Potato", "Corn_(maize)")),
    FarmSite("MP-UJN", "Ujjain", "Madhya Pradesh", 23.1793, 75.7849,
             "Central Plateau & Hills", "Medium black soil",
             ("Soybean", "Corn_(maize)")),
    FarmSite("MH-LTR", "Latur", "Maharashtra", 18.4088, 76.5604,
             "Western Plateau & Hills", "Deep black soil",
             ("Soybean", "Grape")),
    FarmSite("RJ-KOT", "Kota", "Rajasthan", 25.2138, 75.8648,
             "Western Dry Region", "Alluvial-black mixed",
             ("Soybean", "Corn_(maize)")),
    FarmSite("MP-CHW", "Chhindwara", "Madhya Pradesh", 22.0574, 78.9382,
             "Central Plateau & Hills", "Medium black soil",
             ("Corn_(maize)", "Soybean")),

    # --- Southern plateau: maize, tomato, capsicum ---
    FarmSite("KA-KLR", "Kolar", "Karnataka", 13.1362, 78.1291,
             "Southern Plateau & Hills", "Red loamy soil",
             ("Tomato", "Pepper,_bell", "Corn_(maize)")),
    FarmSite("AP-MDP", "Madanapalle", "Andhra Pradesh", 13.5503, 78.5029,
             "Southern Plateau & Hills", "Red loamy soil",
             ("Tomato", "Pepper,_bell")),
    FarmSite("KA-DVG", "Davangere", "Karnataka", 14.4644, 75.9218,
             "Southern Plateau & Hills", "Red sandy loam",
             ("Corn_(maize)", "Tomato")),
    FarmSite("TS-NZB", "Nizamabad", "Telangana", 18.6725, 78.0941,
             "Southern Plateau & Hills", "Red sandy loam",
             ("Corn_(maize)", "Soybean")),
    FarmSite("TS-KRM", "Karimnagar", "Telangana", 18.4386, 79.1288,
             "Southern Plateau & Hills", "Red sandy loam",
             ("Corn_(maize)",)),

    # --- Southern temperate hills: high-altitude berry pockets ---
    FarmSite("TN-OOT", "Udhagamandalam (Ooty)", "Tamil Nadu", 11.4102, 76.6950,
             "Southern Plateau & Hills", "Lateritic hill soil",
             ("Blueberry", "Potato", "Strawberry")),
    FarmSite("KL-MNR", "Munnar", "Kerala", 10.0889, 77.0595,
             "West Coast Plains & Ghats", "Lateritic hill soil",
             ("Blueberry", "Strawberry")),
    FarmSite("KA-CKM", "Chikkamagaluru", "Karnataka", 13.3161, 75.7720,
             "Southern Plateau & Hills", "Lateritic red soil",
             ("Blueberry", "Orange", "Pepper,_bell")),
)


# Growing seasons per crop, as ((start_month, start_day), length_in_days).
# Sampling an observation date inside these windows keeps the climate query
# agronomically sensible: a potato image is never paired with peak-monsoon
# weather, and a kharif maize image always lands in the monsoon.
CROP_SEASONS: dict[str, tuple[tuple[int, int], int]] = {
    "Apple":                    ((4, 1), 180),   # Apr-Sep, flowering to harvest
    "Cherry_(including_sour)":  ((3, 15), 120),  # Mar-Jul
    "Peach":                    ((3, 1), 150),   # Mar-Jul
    "Raspberry":                ((4, 1), 150),   # Apr-Aug
    "Blueberry":                ((3, 1), 180),   # Mar-Aug
    "Strawberry":               ((10, 15), 150), # Oct-Mar, winter crop
    "Grape":                    ((10, 1), 180),  # Oct-Mar, post-monsoon pruning cycle
    "Orange":                   ((6, 15), 180),  # Jun-Dec, Mrig bahar
    "Potato":                   ((10, 15), 120), # Oct-Feb, rabi
    "Tomato":                   ((6, 15), 150),  # Jun-Nov, kharif-rabi overlap
    "Pepper,_bell":             ((7, 1), 150),   # Jul-Nov
    "Squash":                   ((6, 1), 120),   # Jun-Sep, kharif
    "Corn_(maize)":             ((6, 15), 120),  # Jun-Oct, kharif
    "Soybean":                  ((6, 20), 120),  # Jun-Oct, kharif
}


def sites_for_crop(crop: str) -> list[FarmSite]:
    """Every registered site that actually grows ``crop``."""
    return [s for s in SITES if crop in s.crops]


def all_crops() -> list[str]:
    """Sorted list of crops covered by the registry."""
    return sorted({c for s in SITES for c in s.crops})


def site_index() -> dict[str, FarmSite]:
    """Lookup from ``site_id`` to the site record."""
    return {s.site_id: s for s in SITES}


def to_frame(sites: Iterable[FarmSite] = SITES) -> pd.DataFrame:
    """Registry as a DataFrame, one row per site."""
    return pd.DataFrame(
        [
            {
                "site_id": s.site_id, "name": s.name, "state": s.state,
                "lat": s.lat, "lon": s.lon, "agro_zone": s.agro_zone,
                "soil_type": s.soil_type, "crops": "|".join(s.crops),
                "n_crops": len(s.crops),
            }
            for s in sites
        ]
    )
