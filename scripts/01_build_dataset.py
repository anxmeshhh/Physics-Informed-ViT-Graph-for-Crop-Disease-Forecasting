"""Stage 1 - Build the grounded dataset.

    PlantVillage images  +  real Open-Meteo climate  ->  observations table

Run:  python scripts/01_build_dataset.py
Out:  data/processed/observations.parquet
      data/processed/climate_features.parquet
      outputs/reports/stage1_*.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropforecast.config import ensure_dirs, load_config, set_seed          # noqa: E402
from cropforecast.data.assign import (                                      # noqa: E402
    add_risk_levels, assign_observations, assignment_report,
    attach_forecast_targets,
)
from cropforecast.data.climate import fetch_all                             # noqa: E402
from cropforecast.data.farms import to_frame                                # noqa: E402
from cropforecast.data.plantvillage import build_index, class_table         # noqa: E402
from cropforecast.features.agromet import build_climate_features            # noqa: E402
from cropforecast.physics.contagion import simulate_all                     # noqa: E402


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.project.seed)
    reports = Path(cfg.paths.reports)
    processed = Path(cfg.paths.processed)

    print("=" * 78)
    print("STAGE 1  |  Grounded dataset construction")
    print("=" * 78)

    # --- 1. Image index ----------------------------------------------------
    print("\n[1/5] Indexing PlantVillage images ...")
    index = build_index(
        cfg.paths.raw_images, cfg.paths.splits, cfg.paths.leaf_map,
        max_per_class=cfg.data.max_images_per_class,
    )
    print(f"      {len(index):,} images | {index.class_name.nunique()} classes | "
          f"{index.crop.nunique()} crops | {index.leaf_group.nunique():,} leaf groups")
    class_table(index).to_csv(reports / "stage1_class_table.csv", index=False)

    # --- 2. Climate --------------------------------------------------------
    print("\n[2/5] Loading climate archive (cached) ...")
    climate_raw = fetch_all("2020-12-01", "2023-12-31", cfg.paths.climate,
                            daily_vars=list(cfg.climate.daily_vars), verbose=False)
    print(f"      {len(climate_raw):,} site-days across {climate_raw.site_id.nunique()} sites")

    print("\n[3/5] Engineering agro-meteorological features ...")
    climate = build_climate_features(
        climate_raw, windows=tuple(cfg.climate.rolling_windows)
    )
    climate.to_parquet(processed / "climate_features.parquet", index=False)
    print(f"      {climate.shape[1]} columns "
          f"(derived: VPD, leaf wetness, GDD, dew-point depression, wind vector)")

    # --- 3. Epidemic simulation --------------------------------------------
    print("\n[4/6] Simulating wind-borne epidemic spread across the network ...")
    outbreaks = simulate_all(climate, verbose=True)
    outbreaks.to_parquet(processed / "outbreaks.parquet", index=False)
    print(f"      {len(outbreaks):,} crop-site-days simulated")

    # --- 4. Spatio-temporal grounding --------------------------------------
    print("\n[5/6] Assigning each leaf to a real (farm, date) ...")
    obs = assign_observations(
        index, climate,
        years=list(cfg.data.study_years),
        window_days=cfg.data.season_window_days,
        seed=cfg.project.seed,
        outbreaks=outbreaks,
        verbose=True,
    )

    # Join the weather actually recorded at that farm on that date.
    obs = obs.merge(climate, on=["site_id", "date"], how="left", validate="m:1")
    missing = obs.temperature_2m_mean.isna().sum()
    if missing:
        raise RuntimeError(f"{missing} observations failed to join climate")

    # --- 5. Forecast targets ----------------------------------------------
    print("\n[6/6] Attaching multi-horizon forecast targets ...")
    horizons = list(cfg.model.heads.horizons)
    obs = attach_forecast_targets(obs, climate, horizons, outbreaks=outbreaks)
    obs, risk_thresholds = add_risk_levels(obs, horizons, mode='tercile')

    before = len(obs)
    obs = obs.dropna(subset=[f"risk_t{h}" for h in horizons]).reset_index(drop=True)
    print(f"      dropped {before - len(obs)} rows whose horizon ran past the archive")

    obs.to_parquet(processed / "observations.parquet", index=False)
    (processed / "risk_thresholds.json").write_text(
        json.dumps(risk_thresholds, indent=2), encoding="utf-8")
    to_frame().to_csv(reports / "stage1_sites.csv", index=False)

    rep = assignment_report(obs)
    rep.to_csv(reports / "stage1_assignment_report.csv")

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"observations : {len(obs):,}")
    print(f"columns      : {obs.shape[1]}")
    print(f"date range   : {obs.date.min().date()} .. {obs.date.max().date()}")
    print(f"sites used   : {obs.site_id.nunique()}")
    for h in horizons:
        print(f"risk_t{h}  mean={obs[f'risk_t{h}'].mean():.3f}  "
              f"std={obs[f'risk_t{h}'].std():.3f}  "
              f"level split={obs[f'risk_level_t{h}'].value_counts(normalize=True).sort_index().round(3).to_dict()}")
    print("\nClimate separation across classes (coldest / hottest 5):")
    print(pd.concat([rep.head(5), rep.tail(5)])[
        ["n", "sites", "temp_C", "rh_pct", "rain_mm", "lwd_h", "vpd"]].to_string())
    print("\nsaved -> data/processed/observations.parquet")


if __name__ == "__main__":
    main()
