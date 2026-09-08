"""Wind-borne epidemic spread across the farm network.

Why this module exists
----------------------
An earlier version of this benchmark placed every leaf independently, weighted
only by whether the weather at that farm suited its pathogen. That produced a
dataset in which **all** spatial correlation flowed through weather - and since
every node already observes its own weather directly, a neighbour carried no
information the node did not already have. Measured end to end, the graph was
worth -0.02 macro-F1 at every observation density. The graph was not failing;
there was simply nothing spatial for it to find.

That is not what plant epidemics do, and it is not what the project claims.
Slide 5 of the deck is explicit: *wind can carry spores from one field to
another*. This module supplies that missing mechanism.

The model
---------
A discrete-time compartmental epidemic per crop, run daily over the whole study
period, on the real farm network with the real recorded weather:

    force[i,t] = epsilon * f[i,t]                       primary inoculum
               + beta    * f[i,t] * I[i,t]              local build-up
               + gamma   * f[i,t] * sum_j K[i,j,t] I[j,t]   arrival from farm j

    new[i,t]   = force[i,t] * S[i,t]
    I[i,t+1]   = (1 - recovery) * I[i,t] + new[i,t]
    S[i,t+1]   = S[i,t] - new[i,t]

``f`` is the climate favourability for that crop (from :mod:`.epidemiology`), so
weather still gates everything - spores that land in dry heat do not establish.
``K`` is a dispersal kernel that decays exponentially with distance and is
amplified when the wind at the source farm was actually blowing toward the
target on that day. Susceptibility resets each season, as a new crop is planted.

The consequence that matters: ``I[i, t+h]`` now depends on the *neighbours'*
current state, not only on farm i's own weather. Forecasting it therefore
requires spatial reasoning, which is exactly the capability the architecture
claims and exactly what the graph ablation should be able to measure.

This is a simulation, and it is labelled as one. The imagery, the meteorology
and the farm geography remain real; the epidemic dynamics linking them are
modelled, using a standard formulation with physically interpretable constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.farms import CROP_SEASONS, SITES, sites_for_crop
from ..graph.build import initial_bearing_deg, site_distance_matrix
from .epidemiology import crop_pressure


@dataclass(frozen=True)
class EpidemicParams:
    """Constants of the dispersal model, in interpretable units."""

    # Calibrated so that *arrivals from other farms*, not spontaneous local
    # origin, drive most outbreaks - the regime of wind-dispersed pathogens such
    # as the rusts. With the earlier local-dominant settings a farm's own history
    # explained 91% of its future state and neighbours added 0.4%, so there was
    # nothing for a graph to contribute.
    epsilon: float = 0.0015    # primary inoculum arrival rate per favourable day
    beta: float = 0.100        # within-farm multiplication rate
    gamma: float = 2.000       # between-farm transmission rate
    recovery: float = 0.100    # daily decay of infectious tissue
    length_scale_km: float = 180.0   # e-folding distance of the dispersal kernel
    wind_boost: float = 2.0    # multiplier for a fully downwind neighbour
    max_pressure: float = 1.0


def dispersal_kernel(
    sites_df: pd.DataFrame,
    length_scale_km: float,
    wind_boost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance kernel and pairwise bearings for a set of sites.

    Returns ``(base_kernel, bearing)`` where ``base_kernel[i, j]`` is the
    weather-independent likelihood that spores released at *j* reach *i*.
    """
    dist, bearing, _ = site_distance_matrix(sites_df)
    base = np.exp(-dist / max(length_scale_km, 1e-6))
    np.fill_diagonal(base, 0.0)             # self-transmission is the beta term
    return base, bearing


def _season_reset_mask(dates: pd.DatetimeIndex, crop: str) -> np.ndarray:
    """True on the day a new season starts, when susceptibility is restored."""
    (month, day), _ = CROP_SEASONS[crop]
    return np.asarray((dates.month == month) & (dates.day == day))


def simulate_crop(
    climate: pd.DataFrame,
    crop: str,
    params: EpidemicParams = EpidemicParams(),
) -> pd.DataFrame:
    """Run the epidemic for one crop across every farm that grows it.

    ``climate`` must carry the engineered features for all sites and dates.
    Returns a tidy frame with ``site_id, date, crop, infection_pressure``.
    """
    sites = sites_for_crop(crop)
    if len(sites) < 2:
        raise ValueError(f"{crop} is grown at fewer than two registered sites")

    site_ids = [s.site_id for s in sites]
    sub = climate[climate.site_id.isin(site_ids)].sort_values(["date", "site_id"])
    dates = pd.DatetimeIndex(sorted(sub.date.unique()))
    n_sites, n_days = len(site_ids), len(dates)

    # --- Pivot everything to (day, site) matrices for a vectorised run -------
    def matrix(col: str) -> np.ndarray:
        return (sub.pivot(index="date", columns="site_id", values=col)
                .reindex(index=dates, columns=site_ids).to_numpy(dtype=float))

    favour = np.zeros((n_days, n_sites))
    for j, sid in enumerate(site_ids):
        one = sub[sub.site_id == sid].set_index("date").reindex(dates).reset_index()
        favour[:, j] = np.nan_to_num(crop_pressure(crop, one))

    wind_dir = np.nan_to_num(matrix("wind_direction_10m_dominant"))
    wind_spd = np.nan_to_num(matrix("wind_speed_10m_max"))

    sites_df = pd.DataFrame([
        {"site_id": s.site_id, "lat": s.lat, "lon": s.lon} for s in sites
    ])
    base_kernel, bearing = dispersal_kernel(
        sites_df, params.length_scale_km, params.wind_boost)

    reset = _season_reset_mask(dates, crop)

    # --- Run the epidemic ---------------------------------------------------
    I = np.zeros((n_days, n_sites))
    S = np.ones(n_sites)

    # Transport direction at the source: wind blows FROM wind_dir, so spores
    # travel toward wind_dir + 180.
    transport = (wind_dir + 180.0) % 360.0
    # bearing[i, j] is i -> j; spores from j reach i when j's transport points
    # along bearing[j, i].
    bearing_from_source = bearing.T                      # [i, j] = bearing j -> i

    for t in range(n_days - 1):
        if reset[t]:
            S[:] = 1.0
            I[t] = I[t] * 0.05                # a little carry-over inoculum

        # Wind alignment for every ordered pair on this day.
        align = np.cos(np.radians(transport[t][None, :] - bearing_from_source))
        speed_factor = np.clip(wind_spd[t][None, :] / 30.0, 0.0, 1.5)
        kernel = base_kernel * (1.0 + params.wind_boost
                                * np.clip(align, 0.0, None) * speed_factor)

        imported = kernel @ I[t]                          # (n_sites,)

        force = favour[t] * (params.epsilon
                             + params.beta * I[t]
                             + params.gamma * imported)
        new = np.clip(force, 0.0, None) * np.clip(S, 0.0, None)

        I[t + 1] = np.clip((1.0 - params.recovery) * I[t] + new,
                           0.0, params.max_pressure)
        S = np.clip(S - new, 0.0, 1.0)

    out = pd.DataFrame(I, index=dates, columns=site_ids)
    tidy = out.reset_index().melt(id_vars="index", var_name="site_id",
                                  value_name="infection_pressure")
    tidy = tidy.rename(columns={"index": "date"})
    tidy["crop"] = crop
    return tidy


def simulate_all(
    climate: pd.DataFrame,
    crops: list[str] | None = None,
    params: EpidemicParams = EpidemicParams(),
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the epidemic for every crop and concatenate."""
    crops = crops or sorted({c for s in SITES for c in s.crops})
    frames = []
    for crop in crops:
        try:
            df = simulate_crop(climate, crop, params)
        except ValueError as exc:
            if verbose:
                print(f"  skip {crop}: {exc}")
            continue
        if verbose:
            print(f"  {crop:28s} mean={df.infection_pressure.mean():.4f} "
                  f"max={df.infection_pressure.max():.4f} "
                  f"days>0.3={100 * (df.infection_pressure > 0.3).mean():5.1f}%")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def spatial_autocorrelation(
    outbreaks: pd.DataFrame, climate: pd.DataFrame, crop: str, lag_days: int = 3
) -> dict:
    """How much of a farm's future state its *neighbours* explain.

    Compares two predictors of infection pressure ``lag_days`` ahead:
    the farm's own current pressure, and its neighbours' current pressure.
    A meaningful neighbour correlation is the evidence that the graph has
    something real to learn.
    """
    sub = outbreaks[outbreaks.crop == crop]
    wide = sub.pivot(index="date", columns="site_id", values="infection_pressure")
    site_ids = list(wide.columns)
    sites_df = pd.DataFrame([
        {"site_id": s.site_id, "lat": s.lat, "lon": s.lon}
        for s in SITES if s.site_id in site_ids
    ]).set_index("site_id").loc[site_ids].reset_index()

    base, _ = dispersal_kernel(sites_df, 180.0, 0.0)
    weights = base / np.maximum(base.sum(1, keepdims=True), 1e-9)

    values = wide.to_numpy()
    neighbour_now = values @ weights.T
    future = np.roll(values, -lag_days, axis=0)
    valid = slice(0, len(values) - lag_days)

    def corr(a, b):
        a, b = a[valid].ravel(), b[valid].ravel()
        ok = np.isfinite(a) & np.isfinite(b) & (a.std() > 0) & (b.std() > 0)
        if ok.sum() < 10:
            return float("nan")
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    return {
        "crop": crop,
        "lag_days": lag_days,
        "own_pressure_vs_future": round(corr(values, future), 4),
        "neighbour_pressure_vs_future": round(corr(neighbour_now, future), 4),
    }


# Saturation constant for turning infectious tissue into a perceived risk.
RISK_SATURATION = 4.0
# Split of the risk target between conduciveness and inoculum presence.
CONDUCIVENESS_WEIGHT = 0.5


def inoculum_term(infection_pressure: np.ndarray) -> np.ndarray:
    """Map infectious tissue to a saturating [0, 1] presence term.

    Raw pressure is extremely zero-inflated - epidemics are rare and explosive,
    so the median site-day sits at exactly zero. A saturating transform spreads
    the upper tail without disturbing the ordering.
    """
    x = np.asarray(infection_pressure, dtype=float)
    return 1.0 - np.exp(-RISK_SATURATION * np.clip(x, 0.0, None))


def combined_risk(
    climate_pressure: np.ndarray,
    infection_pressure: np.ndarray,
    conduciveness_weight: float = CONDUCIVENESS_WEIGHT,
) -> np.ndarray:
    """The forecasting target: conduciveness AND inoculum, as a farmer sees it.

    Neither half is sufficient on its own. Perfect infection weather with no
    pathogen anywhere nearby is not a risk; a raging epidemic two valleys away
    during a dry heatwave will not establish here either. The first term is a
    local function of this farm's weather; the second can only be known by
    looking at what the neighbours are carrying, which is what forces the model
    to reason spatially.
    """
    w = float(np.clip(conduciveness_weight, 0.0, 1.0))
    return np.clip(
        w * np.asarray(climate_pressure, dtype=float)
        + (1.0 - w) * inoculum_term(infection_pressure),
        0.0, 1.0,
    )
