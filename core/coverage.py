"""Buffer-based pharmacy coverage.

The original report's reward function used network-isochrone polygons only
to *visualize* existing-pharmacy coverage; the actual candidate scoring used
in the reward/MAB step was already a plain distance buffer
(`cand_geom.buffer(804.672)`, i.e. 0.5 miles). This module makes that
buffer-based scoring explicit and configurable (`desert_radius_miles`)
instead of a hardcoded constant, with no behavior regression from the
original model's actual reward computation.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.strtree import STRtree

METERS_PER_MILE = 1609.344
PROJECTED_CRS = "EPSG:3857"


def residential_per_tract(
    tracts: gpd.GeoDataFrame, zoned_land: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Clip residential+commercial land to each tract once, returning one
    unioned geometry + its area per tract (`geoid`, `residential_area`,
    `geometry`), all in a projected CRS so areas/distances are in meters."""
    tracts_p = tracts.to_crs(PROJECTED_CRS)
    land_p = zoned_land.to_crs(PROJECTED_CRS)
    land_p = land_p[land_p["landuse"].isin(["residential", "commercial"])]

    clipped = gpd.overlay(
        tracts_p[["geoid", "geometry"]], land_p[["geometry"]], how="intersection"
    )
    if clipped.empty:
        out = tracts_p[["geoid", "geometry"]].copy()
        out["residential_area"] = out.geometry.area
        return out

    unioned = clipped.dissolve(by="geoid", as_index=False)
    unioned["residential_area"] = unioned.geometry.area

    # Tracts with no residential/commercial OSM tag at all fall back to the
    # full tract polygon, so they aren't silently dropped from the analysis.
    missing = tracts_p[~tracts_p["geoid"].isin(unioned["geoid"])][["geoid", "geometry"]]
    if not missing.empty:
        missing = missing.copy()
        missing["residential_area"] = missing.geometry.area
        unioned = pd.concat([unioned, missing], ignore_index=True)

    return unioned[["geoid", "residential_area", "geometry"]]


class CoverageIndex:
    """Spatial-index-backed coverage lookup so scoring ~300 candidates
    against ~800 tracts doesn't mean 240k unindexed intersection checks."""

    def __init__(self, tract_residential: gpd.GeoDataFrame):
        self.geoids = tract_residential["geoid"].tolist()
        self.geoms = tract_residential.geometry.tolist()
        self.areas = dict(zip(self.geoids, tract_residential["residential_area"]))
        self.tree = STRtree(self.geoms)

    def coverage_fractions(
        self, point_geom, radius_miles: float | dict[str, float]
    ) -> dict[str, float]:
        """Fraction of each nearby tract's residential area covered by a
        buffer around `point_geom` (already in the same projected CRS as the
        tracts used to build this index).

        `radius_miles` is either one fixed radius for every tract, or a
        per-tract `{geoid: radius_miles}` lookup (e.g. from
        `acs.tiered_desert_radius`) — each tract is then measured against
        its *own* threshold, since the tiered desert radius is a property of
        the residential tract, not of the pharmacy.
        """
        if isinstance(radius_miles, dict):
            radius_lookup = radius_miles
            default_radius = max(radius_lookup.values()) if radius_lookup else 0.5
        else:
            radius_lookup = None
            default_radius = radius_miles

        buf_max = point_geom.buffer(default_radius * METERS_PER_MILE)
        candidate_idxs = self.tree.query(buf_max)
        buf_cache = {default_radius: buf_max}
        fractions: dict[str, float] = {}
        for idx in candidate_idxs:
            geoid = self.geoids[idx]
            area = self.areas.get(geoid, 0.0)
            if area <= 0:
                continue
            r = radius_lookup.get(geoid, default_radius) if radius_lookup else default_radius
            if r not in buf_cache:
                buf_cache[r] = point_geom.buffer(r * METERS_PER_MILE)
            overlap = self.geoms[idx].intersection(buf_cache[r])
            if not overlap.is_empty:
                fractions[geoid] = min(overlap.area / area, 1.0)
        return fractions


class BlockCoverageIndex:
    """Population-weighted drop-in replacement for `CoverageIndex`.

    `CoverageIndex` assumes a tract's population is spread uniformly across
    its residential+commercial land, so "fraction covered" is really "area
    fraction covered". This class uses actual 2020 Census block population
    instead: each block contributes its real population, weighted by how
    much of *that block's* area falls inside the buffer, and tract-level
    coverage is the population-weighted sum over its blocks divided by the
    tract's total population. Same `coverage_fractions(point, radius) ->
    {tract_geoid: fraction}` interface as `CoverageIndex`, so callers
    (`coverage_from_points`, `pipeline.run_optimization`) don't need to know
    which one they're using.
    """

    def __init__(self, blocks: gpd.GeoDataFrame):
        self.tract_geoids = blocks["tract_geoid"].tolist()
        self.geoms = blocks.geometry.tolist()
        self.populations = blocks["population"].tolist()
        self.tree = STRtree(self.geoms)
        self.tract_population = blocks.groupby("tract_geoid")["population"].sum().to_dict()

    def coverage_fractions(
        self, point_geom, radius_miles: float | dict[str, float]
    ) -> dict[str, float]:
        if isinstance(radius_miles, dict):
            radius_lookup = radius_miles
            default_radius = max(radius_lookup.values()) if radius_lookup else 0.5
        else:
            radius_lookup = None
            default_radius = radius_miles

        buf_max = point_geom.buffer(default_radius * METERS_PER_MILE)
        candidate_idxs = self.tree.query(buf_max)
        buf_cache = {default_radius: buf_max}
        covered_population: dict[str, float] = {}

        for idx in candidate_idxs:
            tract_geoid = self.tract_geoids[idx]
            pop = self.populations[idx]
            block_geom = self.geoms[idx]
            block_area = block_geom.area
            if pop <= 0 or block_area <= 0:
                continue
            r = radius_lookup.get(tract_geoid, default_radius) if radius_lookup else default_radius
            if r not in buf_cache:
                buf_cache[r] = point_geom.buffer(r * METERS_PER_MILE)
            overlap = block_geom.intersection(buf_cache[r])
            if overlap.is_empty:
                continue
            frac_of_block = overlap.area / block_area
            covered_population[tract_geoid] = covered_population.get(tract_geoid, 0.0) + pop * frac_of_block

        fractions: dict[str, float] = {}
        for tract_geoid, covered_pop in covered_population.items():
            total_pop = self.tract_population.get(tract_geoid, 0.0)
            if total_pop > 0:
                fractions[tract_geoid] = min(covered_pop / total_pop, 1.0)
        return fractions


def nearest_distance_miles(points: gpd.GeoDataFrame, targets: gpd.GeoDataFrame) -> pd.Series:
    """Distance in miles from each point to its nearest target, aligned to
    `points`'s original index. NaN for every point if `targets` is empty
    (e.g. a city with no existing pharmacies at all) rather than raising."""
    if targets.empty:
        return pd.Series([float("nan")] * len(points), index=points.index)
    points_p = points.to_crs(PROJECTED_CRS)
    targets_p = targets.to_crs(PROJECTED_CRS)
    joined = gpd.sjoin_nearest(points_p, targets_p[["geometry"]], distance_col="_dist_m")
    # A tie (equidistant targets) can produce more than one matching row
    # per point -- keep just the first so the result lines up 1:1.
    joined = joined[~joined.index.duplicated(keep="first")]
    return (joined["_dist_m"] / METERS_PER_MILE).reindex(points.index)


def merged_buffer(points: gpd.GeoDataFrame, radius_miles: float):
    """Union of `radius_miles` buffers around every point, in EPSG:3857.
    Returns `None` for an empty input.

    This is the "isochrone" shown on the map: not a real street-network
    travel-time polygon (the original report computed those, but only ever
    used them for visualization -- the actual scoring was always a plain
    distance buffer, same as this app). Showing the literal buffer that the
    reward math uses, instead of a separately-computed network isochrone
    that doesn't match it, means the picture and the math agree.

    Uses one fixed radius rather than a per-tract tiered radius dict --
    tiered radius is a property of the *resident's* tract, not a fixed
    "reach" a pharmacy has, so it doesn't really have a single buffer shape
    to draw. The visualization uses the base radius; the actual coverage
    math (`CoverageIndex`/`BlockCoverageIndex`) still respects tiering.
    """
    if points.empty:
        return None
    points_p = points.to_crs(PROJECTED_CRS)
    return points_p.geometry.buffer(radius_miles * METERS_PER_MILE).union_all()


def coverage_from_points(
    points: gpd.GeoDataFrame,
    index: "CoverageIndex | BlockCoverageIndex",
    radius_miles: float | dict[str, float],
) -> dict[str, float]:
    """Union coverage (1 - product of not-covered) across many points
    (e.g. all existing pharmacies) against the same tract index."""
    if points.empty:
        return {}
    points_p = points.to_crs(PROJECTED_CRS)
    coverage: dict[str, float] = {}
    for geom in points_p.geometry:
        fracs = index.coverage_fractions(geom, radius_miles)
        for geoid, frac in fracs.items():
            prev = coverage.get(geoid, 0.0)
            coverage[geoid] = 1.0 - (1.0 - prev) * (1.0 - frac)
    return coverage
