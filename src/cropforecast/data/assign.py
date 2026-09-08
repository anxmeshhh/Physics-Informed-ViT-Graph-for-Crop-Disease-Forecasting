"""Block 1 - Spatio-temporal grounding of the image set.

PlantVillage tells us *what* disease is on a leaf but not *where* or *when* the
leaf was photographed. Without a location and a date there is no graph, no
climate vector and nothing to forecast, so this module supplies both.

How the grounding works
-----------------------
For every image we choose a real (farm, date) pair such that:

* the farm actually grows that crop (from the registry), and
* the date falls inside that crop's real growing season, and
* the **real measured weather** at that farm on that date was epidemiologically
  plausible for the class on the label.

The third condition is what creates a genuine, learnable relationship between
climate and disease. Crucially the draw is *stochastic*, and a probability floor
guarantees that some images land in unfavourable conditions. The mapping is
therefore noisy and cannot be inverted perfectly - a model that ignores the
image and looks only at the weather will do far better than chance but nowhere
near ceiling, which is exactly the behaviour a real forecasting problem has.

Because neighbouring farms share real weather systems, spatial autocorrelation
of outbreaks emerges on its own; it is never injected by hand.

This is an honest description of a *constructed benchmark*. The imagery and the
meteorology are both real; their pairing is synthetic and is documented as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..physics.contagion import combined_risk, inoculum_term
from ..physics.epidemiology import crop_pressure, favourability
from .farms import CROP_SEASONS, SITES, sites_for_crop

# Minimum selection probability, as a fraction of the mean weight. Higher values
# mean a noisier, harder, more realistic benchmark.
PROB_FLOOR = 0.12
# Sharpness of preference for favourable conditions (1.0 = proportional).
SELECTION_POWER = 1.5


def season_mask(dates: pd.Series, crop: str, window_days: int | None = None) -> np.ndarray:
    """Boolean mask: which dates fall inside ``crop``'s growing season.

    Seasons that run past 31 December (grape, potato, strawberry) wrap correctly
    into the following calendar year.
    """
    (start_month, start_day), length = CROP_SEASONS[crop]
    if window_days is not None:
        length = min(length, window_days)

    dates = pd.to_datetime(dates)
    doy = dates.dt.dayofyear.to_numpy()
    is_leap = dates.dt.is_leap_year.to_numpy()

    start_doy = pd.Timestamp(2021, start_month, start_day).dayofyear
    offset = (doy - start_doy) % np.where(is_leap, 366, 365)
    return offset < length


def build_candidate_pool(
    climate: pd.DataFrame, crop: str, years: list[int], window_days: int | None = None
) -> pd.DataFrame:
    """Every (site, date) at which this crop could plausibly be photographed."""
    site_ids = {s.site_id for s in sites_for_crop(crop)}
    if not site_ids:
        raise ValueError(f"No registered site grows {crop!r}")

    pool = climate[
        climate.site_id.isin(site_ids) & climate.date.dt.year.isin(years)
    ].copy()
    return pool[season_mask(pool.date, crop, window_days)].reset_index(drop=True)


def _inoculum_for_pool(pool: pd.DataFrame, outbreaks: pd.DataFrame,
                       crop: str) -> np.ndarray:
    """Saturating inoculum presence for each candidate (site, date)."""
    sub = outbreaks[outbreaks.crop == crop]
    series = pd.Series(
        sub.infection_pressure.to_numpy(),
        index=pd.MultiIndex.from_arrays([sub.site_id, sub.date]),
    )
    key = pd.MultiIndex.from_arrays([pool.site_id, pool.date])
    return inoculum_term(series.reindex(key).fillna(0.0).to_numpy())


def assign_observations(
    index: pd.DataFrame,
    climate: pd.DataFrame,
    years: list[int],
    window_days: int | None = None,
    seed: int = 42,
    prob_floor: float = PROB_FLOOR,
    power: float = SELECTION_POWER,
    outbreaks: pd.DataFrame | None = None,
    inoculum_weight: float = 0.65,
    verbose: bool = True,
) -> pd.DataFrame:
    """Attach a ``site_id`` and ``date`` to every row of the image index.

    Assignment is done per class so the weighting can use that class's own
    pathogen profile. Images of the *same physical leaf* are kept together at one
    (site, date): they are photographs of one observation event, and splitting
    them across farms would be incoherent.
    """
    rng = np.random.default_rng(seed)
    out_frames: list[pd.DataFrame] = []

    for class_name, group in index.groupby("class_name", sort=True):
        crop = group.crop.iloc[0]
        pool = build_candidate_pool(climate, crop, years, window_days)

        # Weight each candidate day by how plausible this disease was there.
        weight = favourability(class_name, pool).astype(float) ** power

        if outbreaks is not None:
            # A leaf can only be diseased where the pathogen has actually
            # arrived. Weighting by simulated inoculum makes outbreaks cluster
            # in space and time the way real epidemics do, instead of being
            # scattered wherever the weather happened to look suitable.
            inoc = _inoculum_for_pool(pool, outbreaks, crop)
            if class_name.endswith("___healthy"):
                modifier = 1.0 - inoculum_weight * inoc
            else:
                modifier = (1.0 - inoculum_weight) + inoculum_weight * inoc
            weight = weight * np.clip(modifier, 0.0, None)

        weight = weight + prob_floor * max(weight.mean(), 1e-6)
        probs = weight / weight.sum()

        # Draw one (site, date) per LEAF, then broadcast to that leaf's images.
        leaves = group.leaf_group.unique()
        picks = rng.choice(len(pool), size=len(leaves), replace=True, p=probs)
        leaf_to_pick = dict(zip(leaves, picks))

        chosen = pool.iloc[group.leaf_group.map(leaf_to_pick).to_numpy()]
        assigned = group.copy()
        assigned["site_id"] = chosen.site_id.to_numpy()
        assigned["date"] = chosen.date.to_numpy()
        out_frames.append(assigned)

        if verbose:
            print(f"  {class_name:52s} {len(group):5d} imgs  "
                  f"{len(leaves):5d} leaves  pool={len(pool):5d}")

    return pd.concat(out_frames, ignore_index=True).sort_values(
        ["date", "site_id"]).reset_index(drop=True)


def attach_forecast_targets(
    obs: pd.DataFrame,
    climate: pd.DataFrame,
    horizons: list[int],
    outbreaks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add the multi-horizon forecasting targets.

    The target at horizon *h* is the disease risk that farm will actually face
    *h* days later. It has two parts, and needing both is the point:

    * **conduciveness** - whether the weather subsequently recorded there
      favoured infection. This is a local quantity.
    * **inoculum** - whether the pathogen had actually reached that farm, from
      the simulated epidemic. This depends on the *neighbours*, so it cannot be
      predicted from local weather alone.

    Without the second term the target is a pure function of the farm's own
    future weather, and a per-node model matches anything the graph can do.
    """
    climate = climate.sort_values(["site_id", "date"])

    # Pressure depends on the crop, so compute it per crop over the whole archive.
    pressure_lookup: dict[str, pd.Series] = {}
    for crop in obs.crop.unique():
        vals = crop_pressure(crop, climate)
        if outbreaks is not None:
            sub = outbreaks[outbreaks.crop == crop]
            inoc = pd.Series(
                sub.infection_pressure.to_numpy(),
                index=pd.MultiIndex.from_arrays([sub.site_id, sub.date]),
            ).reindex(
                pd.MultiIndex.from_arrays([climate.site_id, climate.date])
            ).fillna(0.0).to_numpy()
            vals = combined_risk(vals, inoc)
        pressure_lookup[crop] = pd.Series(
            vals, index=pd.MultiIndex.from_arrays([climate.site_id, climate.date])
        )

    out = obs.copy()
    for h in horizons:
        target_key = pd.MultiIndex.from_arrays(
            [out.site_id, out.date + pd.Timedelta(days=h)]
        )
        col = np.full(len(out), np.nan)
        for crop, series in pressure_lookup.items():
            m = (out.crop == crop).to_numpy()
            if m.any():
                col[m] = series.reindex(target_key[m]).to_numpy()
        out[f"risk_t{h}"] = col

    # Present-day pressure, for reference and for the physics loss.
    key_now = pd.MultiIndex.from_arrays([out.site_id, out.date])
    col = np.full(len(out), np.nan)
    for crop, series in pressure_lookup.items():
        m = (out.crop == crop).to_numpy()
        if m.any():
            col[m] = series.reindex(key_now[m]).to_numpy()
    out["risk_t0"] = col
    return out


def add_risk_levels(
    obs: pd.DataFrame,
    horizons: list[int],
    mode: str = "tercile",
    lo: float = 0.25,
    hi: float = 0.55,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Discretise each continuous risk into low / medium / high bands.

    ``mode="tercile"`` cuts at the empirical 33rd/67th percentiles. Fixed
    absolute cuts look principled but are not: aggregate pressure is a maximum
    over a crop's pathogens, so it saturates near 1 and a fixed 0.55 threshold
    lands two thirds of all observations in "high", which makes the class almost
    constant and the task trivially easy. Terciles keep the bands informative;
    the resulting thresholds are returned so they can be reported and reused
    unchanged at inference time.
    """
    out = obs.copy()
    thresholds: dict[str, tuple[float, float]] = {}
    for h in horizons:
        col = f"risk_t{h}"
        if mode == "tercile":
            lo_h, hi_h = out[col].quantile([1 / 3, 2 / 3]).tolist()
        else:
            lo_h, hi_h = lo, hi
        thresholds[col] = (float(lo_h), float(hi_h))
        out[f"risk_level_t{h}"] = pd.cut(
            out[col], bins=[-0.01, lo_h, hi_h, 1.01], labels=[0, 1, 2]
        ).astype("Int64")
    return out, thresholds


def assignment_report(obs: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: does the assignment actually separate the classes?

    If the mean favourability-driven weather differs sharply between classes,
    the climate branch has signal to learn.
    """
    return (
        obs.groupby("class_name")
        .agg(
            n=("image_path", "size"),
            sites=("site_id", "nunique"),
            temp_C=("temperature_2m_mean", "mean"),
            rh_pct=("relative_humidity_2m_mean", "mean"),
            rain_mm=("precipitation_sum", "mean"),
            lwd_h=("leaf_wetness_hours", "mean"),
            vpd=("vpd_kpa", "mean"),
        )
        .round(2)
        .sort_values("temp_C")
    )
