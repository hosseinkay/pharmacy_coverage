"""Disk caching for the slow, per-city data layer.

Fetching a city's road network, land-use polygons, etc. from OpenStreetMap
takes seconds-to-minutes. That data barely changes day to day, so it's
cached to disk once per city and reused across app restarts. Everything in
this module is keyed by a "city slug" (e.g. "chicago-illinois-usa") so a
future second city doesn't collide with Chicago's cache.
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def city_cache_dir(city_slug: str) -> Path:
    d = CACHE_DIR / city_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_geodataframe(city_slug: str, key: str, compute_fn) -> gpd.GeoDataFrame:
    """Return a cached GeoDataFrame for (city_slug, key), computing and
    persisting it via compute_fn() on first use."""
    path = city_cache_dir(city_slug) / f"{key}.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    gdf = compute_fn()
    gdf.to_parquet(path)
    return gdf


def cached_dataframe(city_slug: str, key: str, compute_fn) -> pd.DataFrame:
    """Same as `cached_geodataframe` but for a plain (non-geo) DataFrame,
    e.g. an ACS API response."""
    path = city_cache_dir(city_slug) / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = compute_fn()
    df.to_parquet(path)
    return df
