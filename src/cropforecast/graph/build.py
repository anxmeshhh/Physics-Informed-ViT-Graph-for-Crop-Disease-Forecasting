"""Block 2c - Graph construction.

Implements the diagram's pipeline exactly:

    Combine Location + Time  ->  KNN + Radius Search  ->  Spatial Graph

A node is one **observation** (a leaf photographed at a farm on a date). Two
observations are joined when they are close in space *and* close in time, since
an outbreak 400 km away or four months earlier tells you nothing useful about
this field today.

Two edge types are produced:

``proximity``
    Undirected. Nearby farms observed at nearby times; this is the KNN + radius
    component and it carries the ordinary spatial smoothness of an epidemic.

``downwind``
    Directed, and only when ``wind_aware`` is on. If the wind measured at the
    source farm on the source date was actually blowing towards the target farm,
    an extra edge is added in that direction. This is the graph encoding of
    wind-borne spore transport - the mechanism named on slide 5 of the deck.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between two arrays of coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def initial_bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Compass bearing from point 1 to point 2, in degrees clockwise from north."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def site_distance_matrix(sites_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pairwise site distances (km) and bearings (deg). Returns (dist, bearing, ids)."""
    ids = sites_df.site_id.tolist()
    lat = sites_df.lat.to_numpy(dtype=float)
    lon = sites_df.lon.to_numpy(dtype=float)
    n = len(ids)
    dist = np.zeros((n, n))
    bearing = np.zeros((n, n))
    for i in range(n):
        dist[i] = haversine_km(lat[i], lon[i], lat, lon)
        bearing[i] = initial_bearing_deg(lat[i], lon[i], lat, lon)
    return dist, bearing, ids


@dataclass
class SpatioTemporalGraph:
    """Edge list plus edge attributes for one built graph."""

    edge_index: np.ndarray      # (2, E) int64, [source; target]
    edge_attr: np.ndarray       # (E, 4) float32
    edge_type: np.ndarray       # (E,) 0 = proximity, 1 = downwind
    num_nodes: int

    ATTR_NAMES = ("dist_km_norm", "dt_days_norm", "wind_alignment", "is_downwind")

    def summary(self) -> dict:
        deg = np.bincount(self.edge_index[1], minlength=self.num_nodes)
        return {
            "num_nodes": self.num_nodes,
            "num_edges": int(self.edge_index.shape[1]),
            "proximity_edges": int((self.edge_type == 0).sum()),
            "downwind_edges": int((self.edge_type == 1).sum()),
            "mean_in_degree": float(deg.mean()),
            "max_in_degree": int(deg.max()) if self.num_nodes else 0,
            "isolated_nodes": int((deg == 0).sum()),
        }


def build_graph(
    obs: pd.DataFrame,
    sites_df: pd.DataFrame,
    k_neighbours: int = 8,
    radius_km: float = 350.0,
    time_window_days: int = 7,
    max_edges_per_node: int = 16,
    wind_aware: bool = True,
    wind_alignment_threshold: float = 0.5,
    min_neighbours: int = 3,
    same_crop: bool = True,
    seed: int = 42,
) -> SpatioTemporalGraph:
    """Build the spatio-temporal graph over observation rows.

    ``obs`` must be indexed 0..N-1 in the same order as the model's node features
    and must carry ``site_id``, ``date``, ``crop`` and the dominant wind direction.

    ``same_crop`` restricts edges to observations of the same crop, and it should
    normally stay on. Apple scab does not travel from an orchard to a maize
    field, so a cross-crop edge carries no epidemiological information - it only
    mixes unrelated representations together and drives the over-smoothing that
    makes a deep GNN worse than no graph at all. It also cuts the edge count by
    roughly an order of magnitude, which is what keeps GAT inside 6 GB of VRAM.
    """
    rng = np.random.default_rng(seed)
    obs = obs.reset_index(drop=True)
    n_nodes = len(obs)

    dist, bearing, ids = site_distance_matrix(sites_df)
    site_pos = {sid: i for i, sid in enumerate(ids)}

    # --- Radius search, then KNN within it, done once per site ---------------
    # The radius is the agronomic constraint; the KNN fallback is the structural
    # one. India's production belts are far apart, so a pure radius rule strands
    # outlying sites (Munnar, Hooghly) with no neighbours at all and the GNN can
    # never reach them. Guaranteeing `min_neighbours` nearest sites keeps the
    # graph connected while the radius still governs everything else.
    neighbours: dict[str, list[tuple[str, float, float]]] = {}
    for sid, i in site_pos.items():
        order = [j for j in np.argsort(dist[i]) if j != i]
        within = [j for j in order if dist[i, j] <= radius_km][:k_neighbours]
        if len(within) < min_neighbours:
            within = order[:min_neighbours]
        neighbours[sid] = [
            (ids[j], float(dist[i, j]), float(bearing[i, j])) for j in within
        ]

    # --- Bucket observations by (site, day[, crop]) for O(1) temporal lookup --
    day = (obs.date - obs.date.min()).dt.days.to_numpy()
    crops = (obs.crop.to_numpy() if same_crop and "crop" in obs.columns
             else np.full(len(obs), "*", dtype=object))
    buckets: dict[tuple, list[int]] = {}
    for row_i, (sid, d, cr) in enumerate(zip(obs.site_id.to_numpy(), day, crops)):
        buckets.setdefault((sid, int(d), cr), []).append(row_i)

    wind_dir = (obs["wind_direction_10m_dominant"].to_numpy(dtype=float)
                if wind_aware and "wind_direction_10m_dominant" in obs.columns
                else None)

    src_list: list[int] = []
    dst_list: list[int] = []
    attr_list: list[tuple[float, float, float, float]] = []
    type_list: list[int] = []

    site_ids = obs.site_id.to_numpy()
    for tgt in range(n_nodes):
        sid = site_ids[tgt]
        d0 = int(day[tgt])
        crop = crops[tgt]

        candidates: list[tuple[int, float, float, float]] = []  # (src, dist, dt, align)
        for nb_sid, nb_dist, nb_bearing in neighbours[sid]:
            for dd in range(-time_window_days, time_window_days + 1):
                for src in buckets.get((nb_sid, d0 + dd, crop), ()):
                    align = 0.0
                    if wind_dir is not None:
                        # Wind direction is where the wind comes FROM, so the
                        # transport direction is wind_dir + 180. An edge is
                        # "downwind" when transport at the SOURCE points at us.
                        transport = (wind_dir[src] + 180.0) % 360.0
                        # bearing from source site to this target site
                        back_bearing = (nb_bearing + 180.0) % 360.0
                        align = float(np.cos(np.radians(transport - back_bearing)))
                    candidates.append((src, nb_dist, abs(dd), align))

        if not candidates:
            continue

        # Rank by a combined space-time proximity, keeping the closest.
        arr_dist = np.array([c[1] for c in candidates])
        arr_dt = np.array([c[2] for c in candidates], dtype=float)
        cost = (arr_dist / max(radius_km, 1e-6)) ** 2 + \
               (arr_dt / max(time_window_days, 1e-6)) ** 2
        order = np.argsort(cost)
        if len(order) > max_edges_per_node:
            order = order[:max_edges_per_node]

        for oi in order:
            src, dkm, dt, align = candidates[oi]
            if src == tgt:
                continue
            downwind = bool(wind_aware and align >= wind_alignment_threshold)
            src_list.append(src)
            dst_list.append(tgt)
            attr_list.append((dkm / radius_km, dt / max(time_window_days, 1),
                              align, float(downwind)))
            type_list.append(1 if downwind else 0)

    if not src_list:   # degenerate case (e.g. a single-site subset)
        return SpatioTemporalGraph(
            np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.int64), n_nodes,
        )

    return SpatioTemporalGraph(
        edge_index=np.vstack([np.array(src_list), np.array(dst_list)]).astype(np.int64),
        edge_attr=np.array(attr_list, dtype=np.float32),
        edge_type=np.array(type_list, dtype=np.int64),
        num_nodes=n_nodes,
    )
