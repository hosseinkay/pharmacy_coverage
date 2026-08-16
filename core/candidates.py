"""Candidate pharmacy site generation.

The original model sourced candidate sites from a manually-curated LoopNet
export (`candidates.csv`) — real, but Chicago-specific and not reproducible
for another city without repeating that manual research. This module
generates candidates programmatically from open data (a grid sampled over
OSM residential/commercial zoned land), which is what actually makes
multi-city support feasible later. A CSV of real listings can still be
supplied as a higher-fidelity override via `load_candidates_from_csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

PROJECTED_CRS = "EPSG:3857"
METERS_PER_MILE = 1609.344


def generate_grid_candidates(
    zoned_land: gpd.GeoDataFrame, spacing_meters: float = 150.0
) -> gpd.GeoDataFrame:
    """Sample a regular grid of points clipped to residential/commercial land."""
    land = zoned_land.to_crs(PROJECTED_CRS)
    land = land[land["landuse"].isin(["residential", "commercial"])]
    if land.empty:
        return gpd.GeoDataFrame(
            {"candidate_id": [], "label": []}, geometry=[], crs=PROJECTED_CRS
        )

    minx, miny, maxx, maxy = land.total_bounds
    xs = np.arange(minx, maxx, spacing_meters)
    ys = np.arange(miny, maxy, spacing_meters)
    pts = [Point(x, y) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=pts, crs=PROJECTED_CRS)

    # Spatial-join against the individual parcels (STRtree-backed) instead of
    # testing each point against one giant unioned polygon: testing ~tens of
    # thousands of points against an unindexed multi-thousand-vertex union is
    # an unindexed-against-complex-geometry trap that takes many minutes;
    # the join below does the same filtering in well under a second.
    matched_idx = gpd.sjoin(
        grid, land[["geometry"]], predicate="intersects", how="inner"
    ).index.unique()
    within = grid.loc[matched_idx].reset_index(drop=True)
    within["candidate_id"] = within.index
    within["label"] = within["candidate_id"].apply(lambda i: f"Candidate site #{i}")
    return within[["candidate_id", "label", "geometry"]]


def load_candidates_from_csv(
    path: str, lat_col: str = "Latitude", lon_col: str = "Longitude"
) -> gpd.GeoDataFrame:
    """Load a curated candidate list (e.g. real commercial listings)."""
    df = pd.read_csv(path)
    label_col = next(
        (c for c in df.columns if "address" in c.lower()), None
    )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    gdf["candidate_id"] = gdf.index
    gdf["label"] = df[label_col] if label_col else gdf["candidate_id"].astype(str)
    return gdf[["candidate_id", "label", "geometry"]]


def exclude_near_existing(
    candidates: gpd.GeoDataFrame,
    pharmacies: gpd.GeoDataFrame,
    min_distance_miles: float,
) -> gpd.GeoDataFrame:
    """Drop candidates that are already within `min_distance_miles` of an
    existing pharmacy — no point siting a new one right on top of one."""
    if pharmacies.empty or candidates.empty:
        return candidates

    pharm_p = pharmacies.to_crs(PROJECTED_CRS)
    pharm_union = pharm_p.geometry.union_all()
    min_dist_m = min_distance_miles * METERS_PER_MILE

    keep_mask = candidates.geometry.apply(lambda g: g.distance(pharm_union) >= min_dist_m)
    return candidates[keep_mask].reset_index(drop=True)
