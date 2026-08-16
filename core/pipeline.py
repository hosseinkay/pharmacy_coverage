"""Orchestrates the full run, split into two phases so a UI can show the
need index *before* committing to running the siting optimization:

  prepare_city()     load city data -> need index -> clip to residential
                      land (cheap, no candidates/optimization yet)
  run_optimization()  candidates -> coverage -> greedy selection ->
                      before/after results (reuses prepare_city's output)

Both phases, and the standalone `run()` convenience wrapper, are plain
functions with no Streamlit dependency -- the whole methodology is
runnable and testable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from . import acs
from . import acs_need_factors
from . import cache
from . import candidates as cand_mod
from . import census_blocks
from . import chicago_community_areas
from . import coverage as cov_mod
from . import need_index
from . import opportunity_zones
from . import optimize
from .city_data import CityDataBundle, get_data_source
from .config import CandidateConfig, NeedWeights, OptConfig


def _get_need_factors(city_key: str, tract_geoids: list[str]) -> tuple[pd.DataFrame | None, str]:
    """ACS need factors (vehicle access, poverty, age 65+, mobility,
    uninsured, race/ethnicity), cached per city. Returns (df_or_None,
    status_message); a missing key or failed request degrades to "chronic
    medication burden only" rather than breaking the need index."""
    if not acs.get_census_api_key():
        return None, "No CENSUS_API_KEY set -- need index uses chronic medication burden (CDC PLACES) only"
    try:
        slug = cache.slugify(city_key)
        df = cache.cached_dataframe(
            slug, "acs_need_factors", lambda: acs_need_factors.fetch_need_factors(tract_geoids)
        )
        return df, "Need index includes vehicle access, poverty, age 65+, and mobility (ACS) alongside chronic medication burden (CDC PLACES)"
    except acs_need_factors.NeedFactorsUnavailable as exc:
        return None, f"ACS need factors unavailable ({exc}) -- need index uses chronic medication burden (CDC PLACES) only"


@dataclass
class PreparedCity:
    city_key: str
    bundle: CityDataBundle

    tracts: gpd.GeoDataFrame
    """Projected CRS (EPSG:3857). Full tract polygons merged with need
    scores -- kept for population/need lookups, not for display (see
    `tract_residential` for that)."""

    tract_residential: gpd.GeoDataFrame
    """Projected CRS (EPSG:3857). Each tract clipped to its
    residential+commercial OSM land, with need scores joined on. This is
    what should actually be shown on a map: shading the full census tract
    polygon implies need is uniform across land nobody lives on (parks,
    industrial, water), which it isn't."""

    acs_factors: pd.DataFrame | None
    """geoid, no_vehicle_pct, poverty_pct, age65_pct,
    ambulatory_disability_pct, uninsured_pct, black_share, hispanic_share.
    None if ACS was unavailable. race/ethnicity shares are carried through
    purely for the post-hoc equity check in run_optimization's summary --
    never used as a scoring input."""

    need_factors_status: str


def prepare_city(
    city_key: str, need_weights: NeedWeights, use_health_priorities: bool = False
) -> PreparedCity:
    source = get_data_source(city_key)
    bundle = source.load()

    tract_geoids = bundle.tracts["geoid"].tolist()
    if use_health_priorities:
        acs_factors, need_factors_status = _get_need_factors(city_key, tract_geoids)
    else:
        acs_factors, need_factors_status = None, ""

    need_df = need_index.compute_need_scores(
        bundle.tracts,
        bundle.places_scores,
        acs_factors,
        need_weights.normalized(),
        use_health_priorities=use_health_priorities,
    )
    tracts = bundle.tracts.merge(need_df, on="geoid", how="left", suffixes=("", "_dup"))
    tracts = tracts.to_crs(cov_mod.PROJECTED_CRS)

    tract_residential = cov_mod.residential_per_tract(tracts, bundle.zoned_land)
    tract_residential = tract_residential.merge(
        tracts[["geoid", "normalized_need", "weighted_need", "population", "TotalPopulation"]],
        on="geoid",
        how="left",
    )

    return PreparedCity(
        city_key=city_key,
        bundle=bundle,
        tracts=tracts,
        tract_residential=tract_residential,
        acs_factors=acs_factors,
        need_factors_status=need_factors_status,
    )


def rebind_need_scores(
    prepared: PreparedCity,
    weights_dict: dict,
    use_health_priorities: bool,
) -> PreparedCity:
    """Return a PreparedCity identical to `prepared` but with need scores
    recomputed from `weights_dict`.  All geometry / coverage objects are
    reused unchanged — only `weighted_need` and `normalized_need` are
    replaced, making this cheap enough to call once per comparison strategy."""
    need_df = need_index.compute_need_scores(
        prepared.bundle.tracts,
        prepared.bundle.places_scores,
        prepared.acs_factors,
        weights_dict,
        use_health_priorities=use_health_priorities,
    )
    tracts_new = prepared.bundle.tracts.merge(
        need_df, on="geoid", how="left", suffixes=("", "_dup")
    ).to_crs(cov_mod.PROJECTED_CRS)

    need_cols = ["geoid", "normalized_need", "weighted_need", "population", "TotalPopulation"]
    tract_res_new = prepared.tract_residential.drop(
        columns=["normalized_need", "weighted_need"], errors="ignore"
    ).merge(tracts_new[need_cols], on="geoid", how="left")

    return PreparedCity(
        city_key=prepared.city_key,
        bundle=prepared.bundle,
        tracts=tracts_new,
        tract_residential=tract_res_new,
        acs_factors=prepared.acs_factors,
        need_factors_status=prepared.need_factors_status,
    )


@dataclass
class PipelineResult:
    tracts: gpd.GeoDataFrame
    """One row per census tract (EPSG:4326): geoid, population,
    normalized_need, coverage_before_pct, coverage_after_pct, growth_pct."""

    tract_residential: gpd.GeoDataFrame
    """Same coverage/need columns as `tracts`, but geometry is clipped to
    residential+commercial land (EPSG:4326) -- use this for map shading."""

    candidates: gpd.GeoDataFrame
    """All evaluated candidate sites (EPSG:4326): candidate_id, label,
    selected (bool), reward (NaN if not selected), geometry."""

    pharmacies: gpd.GeoDataFrame
    """Existing pharmacies (EPSG:4326): name, geometry."""

    community_impact: pd.DataFrame | None
    """Community-area-level rollup (the public-facing geography): coverage
    before/after/gain, newly covered population, which selected sites'
    service areas touch it, majority-Black/Latino flags. None for cities
    without community-area data."""

    summary: dict
    center: tuple[float, float]


def _resolve_desert_radius(
    city_key: str, tracts: gpd.GeoDataFrame, opt: OptConfig
) -> tuple[float | dict, str]:
    """Returns (radius, status_message). `radius` is a fixed float unless
    `opt.use_tiered_radius` is set and ACS data is actually available, in
    which case it's a per-geoid dict from `acs.tiered_desert_radius`."""
    if not opt.use_tiered_radius:
        return opt.desert_radius_miles, "Fixed radius"

    if not acs.get_census_api_key():
        return (
            opt.desert_radius_miles,
            "Tiered radius requested but no CENSUS_API_KEY is set — using fixed radius",
        )

    try:
        slug = cache.slugify(city_key)
        acs_df = cache.cached_dataframe(
            slug, "acs_tract_data", lambda: acs.fetch_tract_acs(tracts["geoid"].tolist())
        )
        radius_lookup = acs.tiered_desert_radius(
            acs_df,
            base_radius_miles=opt.desert_radius_miles,
            extended_radius_miles=opt.extended_radius_miles,
            low_income_pct_of_median=opt.low_income_pct_of_median,
            max_no_vehicle_share=opt.max_no_vehicle_share,
        )
        extended_count = sum(
            1 for r in radius_lookup.values() if r == opt.extended_radius_miles
        )
        return radius_lookup, f"Tiered radius active — {extended_count} tracts extended to {opt.extended_radius_miles}mi"
    except acs.AcsUnavailable as exc:
        return opt.desert_radius_miles, f"Tiered radius failed ({exc}) — using fixed radius"


def _resolve_coverage_index(
    city_key: str, prepared: PreparedCity, opt: OptConfig
) -> tuple[object, str]:
    """Returns (coverage_index, status_message). Defaults to the area-based
    `CoverageIndex` (assumes uniform population density across a tract's
    residential+commercial land). If `opt.use_population_weighted_coverage`
    is set and Census block data is actually available, returns a
    `BlockCoverageIndex` instead, which weights by real population."""
    area_index = cov_mod.CoverageIndex(prepared.tract_residential)

    if not opt.use_population_weighted_coverage:
        return area_index, "Coverage weighted by land area (assumes population is spread evenly across residential+commercial land)"

    if not acs.get_census_api_key():
        return (
            area_index,
            "Population-weighted coverage requested but no CENSUS_API_KEY is set — using area-based coverage",
        )

    try:
        slug = cache.slugify(city_key)
        bbox = tuple(prepared.tracts.to_crs("EPSG:4326").total_bounds)
        tract_geoids = prepared.tracts["geoid"].tolist()
        blocks = cache.cached_geodataframe(
            slug,
            "census_blocks",
            lambda: census_blocks.get_population_blocks(tract_geoids, bbox),
        )
        return (
            cov_mod.BlockCoverageIndex(blocks),
            f"Coverage weighted by real 2020 Census block population ({len(blocks):,} blocks)",
        )
    except census_blocks.CensusBlocksUnavailable as exc:
        return area_index, f"Population-weighted coverage failed ({exc}) — using area-based coverage"


def _get_community_areas(city_key: str) -> gpd.GeoDataFrame | None:
    """Chicago-only enrichment (see chicago_community_areas.py); returns
    None for any other city rather than raising, so callers can treat it
    as "not available here" instead of a special case to catch."""
    if city_key != "chicago":
        return None
    try:
        slug = cache.slugify(city_key)
        return cache.cached_geodataframe(
            slug, "community_areas", chicago_community_areas.fetch_community_areas
        )
    except chicago_community_areas.CommunityAreasUnavailable:
        return None


def _get_opportunity_zone_tracts(city_key: str, tract_geoids: list[str]) -> set[str]:
    """Designated-OZ tract GEOIDs for this city's tracts. Generalizes to any
    city (OZ designation is itself a tract attribute); returns an empty set
    -- not an error -- if the lookup fails, so a flaky request degrades to
    "nothing flagged" rather than breaking the run."""
    try:
        slug = cache.slugify(city_key)
        df = cache.cached_dataframe(
            slug,
            "opportunity_zone_tracts",
            lambda: pd.DataFrame({"geoid": sorted(opportunity_zones.fetch_designated_tracts(tract_geoids))}),
        )
        return set(df["geoid"])
    except opportunity_zones.OpportunityZonesUnavailable:
        return set()


def _assign_point_in_tract(points: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame) -> dict:
    """candidate_id (or any id column the caller's `points` has) -> the
    geoid of the tract that point physically sits in. One spatial join,
    reused for the Opportunity Zone filter/flag and the candidate's own
    community-area label -- both are properties of *where the site is*,
    not of how far its service buffer reaches."""
    id_col = "candidate_id" if "candidate_id" in points.columns else points.columns[0]
    joined = gpd.sjoin(points[[id_col, "geometry"]], tracts[["geoid", "geometry"]], how="left", predicate="within")
    return dict(zip(joined[id_col], joined["geoid"]))


def _filter_to_opportunity_zones(
    candidates: gpd.GeoDataFrame, candidate_tract: dict, oz_tracts: set[str]
) -> gpd.GeoDataFrame:
    """Keep only candidates physically located inside a designated
    Opportunity Zone tract -- a policy CONSTRAINT on eligibility, applied
    before scoring, not a factor the need index sees. If no tracts in this
    city are OZ-designated (lookup failed or genuinely none), this would
    zero out every candidate; better to leave the candidate set untouched
    and let the caller know nothing was excluded than silently produce an
    empty siting run."""
    if not oz_tracts:
        return candidates
    keep_ids = {cid for cid, geoid in candidate_tract.items() if geoid in oz_tracts}
    return candidates[candidates["candidate_id"].isin(keep_ids)].reset_index(drop=True)


def _compute_equity_summary(tracts_out: gpd.GeoDataFrame, acs_factors: pd.DataFrame | None) -> dict:
    """Post-hoc equity check, *not* a scoring input: what share of the
    newly-covered population lives in majority-Black or majority-Latino
    tracts? Race/ethnicity never influences which sites get picked --
    this only reports on the outcome, asked after the fact rather than
    baked into the optimization."""
    if acs_factors is None or "black_share" not in acs_factors.columns:
        return {"equity_available": False}

    merged = tracts_out.merge(
        acs_factors[["geoid", "black_share", "hispanic_share"]], on="geoid", how="left"
    )
    newly_covered = (
        (merged["coverage_after_pct"] - merged["coverage_before_pct"]).clip(lower=0)
        / 100.0
        * merged["TotalPopulation"]
    )
    total_new = float(newly_covered.sum())
    if total_new <= 0:
        return {"equity_available": True, "equity_pct_majority_black": 0.0, "equity_pct_majority_hispanic": 0.0}

    majority_black = merged["black_share"] > 0.5
    majority_hispanic = merged["hispanic_share"] > 0.5
    return {
        "equity_available": True,
        "equity_pct_majority_black": float(newly_covered[majority_black].sum() / total_new * 100.0),
        "equity_pct_majority_hispanic": float(newly_covered[majority_hispanic].sum() / total_new * 100.0),
    }


def _attach_tooltip_factors(df: pd.DataFrame, prepared: PreparedCity) -> pd.DataFrame:
    """Merge the raw (non-percentile-ranked) factor values onto `df` (must
    have a `geoid` column) for human-readable map tooltips. The need index
    itself uses percentile ranks internally (see need_index.py), but "38%
    of households have no vehicle" means more to a reader than "this tract
    is at the 72nd percentile for vehicle access.\""""
    out = df.copy()
    if prepared.acs_factors is not None:
        cols = ["geoid", "no_vehicle_pct", "poverty_pct", "age65_pct", "ambulatory_disability_pct", "uninsured_pct"]
        out = out.merge(prepared.acs_factors[cols], on="geoid", how="left")
    chronic = (
        prepared.bundle.places_scores[
            prepared.bundle.places_scores["MeasureId"].isin(need_index.CHRONIC_BURDEN_MEASURES)
        ]
        .groupby("geoid")["Data_Value"]
        .mean()
        .rename("chronic_burden_pct")
    )
    return out.merge(chronic, on="geoid", how="left")


def get_preview_layer(prepared: PreparedCity, radius_miles: float) -> gpd.GeoDataFrame:
    """Everything the *pre-optimization* need map needs for rich tooltips:
    community area, Opportunity Zone status, raw factor values, and
    current coverage/nearest-pharmacy-distance at the given radius --
    without generating candidates or running the optimizer at all. Returns
    `tract_residential` (EPSG:4326) with those columns attached."""
    bundle = prepared.bundle
    tracts = prepared.tracts

    cov_index = cov_mod.CoverageIndex(prepared.tract_residential)
    existing_coverage = cov_mod.coverage_from_points(bundle.pharmacies, cov_index, radius_miles)

    # NOTE: `prepared.tract_residential` is *not* in the same row order as
    # `tracts` (the residential clip dissolves by geoid, which reorders
    # rows) -- every value computed against `tracts`/its centroids must be
    # joined back onto `out` by geoid (a dict + .map()), never by raw
    # positional `.values`, or rows silently get the wrong tract's data.
    tract_centroids_3857 = gpd.GeoDataFrame(
        tracts[["geoid"]], geometry=tracts.geometry.centroid, crs=tracts.crs
    )
    nearest = cov_mod.nearest_distance_miles(tract_centroids_3857, bundle.pharmacies)
    nearest_by_geoid = dict(zip(tract_centroids_3857["geoid"], nearest))

    out = prepared.tract_residential.copy()
    out["coverage_pct"] = out["geoid"].map(existing_coverage).fillna(0.0) * 100.0
    out["is_covered"] = out["coverage_pct"] >= 50.0
    out["nearest_pharmacy_miles"] = out["geoid"].map(nearest_by_geoid)

    community_areas = _get_community_areas(prepared.city_key)
    if community_areas is not None:
        centroids_4326 = tract_centroids_3857.to_crs("EPSG:4326")
        joined = gpd.sjoin(centroids_4326, community_areas[["community_area", "geometry"]], how="left", predicate="within")
        community_by_geoid = dict(zip(joined["geoid"], joined["community_area"]))
        out["community_area"] = out["geoid"].map(community_by_geoid)
    else:
        out["community_area"] = None

    oz_tracts = _get_opportunity_zone_tracts(prepared.city_key, tracts["geoid"].tolist())
    out["is_opportunity_zone"] = out["geoid"].isin(oz_tracts)

    out = _attach_tooltip_factors(out, prepared)
    return out.to_crs("EPSG:4326")


def _compute_community_impact(
    tracts_out: gpd.GeoDataFrame,
    candidates_out: gpd.GeoDataFrame,
    acs_factors: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Community-area-level rollup of tract-level results. Community areas
    are the public-facing geography (recognizable names, not GEOIDs);
    tracts are an implementation detail underneath this. Returns None for
    cities without community-area data."""
    if tracts_out["community_area"].isna().all():
        return None

    df = tracts_out.copy()
    df["newly_covered"] = (
        (df["coverage_after_pct"] - df["coverage_before_pct"]) / 100.0 * df["TotalPopulation"]
    ).clip(lower=0)
    if acs_factors is not None and "black_share" in acs_factors.columns:
        df = df.merge(acs_factors[["geoid", "black_share", "hispanic_share"]], on="geoid", how="left")

    def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
        weights = weights.fillna(0)
        total = weights.sum()
        return float((values * weights).sum() / total) if total > 0 else 0.0

    rows = []
    for community, group in df.groupby("community_area"):
        row = {
            "community_area": community,
            "population": float(group["TotalPopulation"].sum()),
            "coverage_before_pct": _weighted_avg(group["coverage_before_pct"], group["TotalPopulation"]),
            "coverage_after_pct": _weighted_avg(group["coverage_after_pct"], group["TotalPopulation"]),
            "newly_covered_population": float(group["newly_covered"].sum()),
        }
        row["coverage_gain_pct"] = row["coverage_after_pct"] - row["coverage_before_pct"]
        if "black_share" in group.columns:
            row["majority_black"] = _weighted_avg(group["black_share"], group["TotalPopulation"]) > 0.5
            row["majority_hispanic"] = _weighted_avg(group["hispanic_share"], group["TotalPopulation"]) > 0.5
        rows.append(row)
    impact = pd.DataFrame(rows)

    selected = candidates_out[candidates_out["selected"]]
    ranks_by_community: dict[str, list[int]] = {}
    for _, site in selected.iterrows():
        names = [n for n in (site.get("neighborhoods") or "").split(", ") if n]
        if not names and site.get("community_area"):
            names = [site["community_area"]]
        for name in names:
            ranks_by_community.setdefault(name, []).append(int(site["rank"]))

    impact["selected_site_ranks"] = impact["community_area"].map(
        lambda c: ", ".join(f"#{r}" for r in sorted(ranks_by_community.get(c, [])))
    )

    return impact.sort_values("coverage_gain_pct", ascending=False).reset_index(drop=True)


def _reached_per_site(
    selected_ids: list,
    candidate_coverage: dict,
    existing_coverage: dict,
    population: dict,
) -> dict:
    """Marginal amount of `population` covered by each selected site, in
    selection order -- the same incremental discounting `optimize.greedy_select`
    does for weighted need, just tracked separately so it can be reported
    per site (total population reached, or just high-need population
    reached) without conflating different units with the need-index reward.
    """
    remaining = {
        geoid: population.get(geoid, 0.0) * (1.0 - existing_coverage.get(geoid, 0.0))
        for geoid in population
    }
    reached = {}
    for cid in selected_ids:
        gain = sum(
            remaining.get(geoid, 0.0) * frac
            for geoid, frac in candidate_coverage[cid].items()
        )
        reached[cid] = gain
        for geoid, frac in candidate_coverage[cid].items():
            if geoid in remaining:
                remaining[geoid] *= 1.0 - frac
    return reached


def run_optimization(
    prepared: PreparedCity, opt: OptConfig, candidate_cfg: CandidateConfig
) -> PipelineResult:
    bundle = prepared.bundle
    tracts = prepared.tracts

    cov_index, coverage_status = _resolve_coverage_index(prepared.city_key, prepared, opt)
    radius, radius_status = _resolve_desert_radius(prepared.city_key, tracts, opt)

    existing_coverage = cov_mod.coverage_from_points(bundle.pharmacies, cov_index, radius)

    oz_tracts = _get_opportunity_zone_tracts(prepared.city_key, tracts["geoid"].tolist())

    if candidate_cfg.source == "csv" and candidate_cfg.csv_path:
        all_candidates = cand_mod.load_candidates_from_csv(candidate_cfg.csv_path)
    else:
        all_candidates = cand_mod.generate_grid_candidates(
            bundle.zoned_land, candidate_cfg.grid_spacing_meters
        )
    all_candidates = cand_mod.exclude_near_existing(
        all_candidates, bundle.pharmacies, opt.min_distance_miles
    )

    # Where each candidate's own point physically sits -- used for the OZ
    # filter/flag and the candidate's own community-area label, both
    # properties of the site itself rather than its service area.
    candidate_tract = _assign_point_in_tract(all_candidates, tracts)

    if opt.restrict_to_opportunity_zones:
        all_candidates = _filter_to_opportunity_zones(all_candidates, candidate_tract, oz_tracts)

    candidate_coverage = {
        cid: cov_index.coverage_fractions(geom, radius)
        for cid, geom in zip(all_candidates["candidate_id"], all_candidates.geometry)
    }

    need_scores = dict(zip(tracts["geoid"], tracts["weighted_need"].fillna(0.0)))

    selection = optimize.greedy_select(
        all_candidates,
        candidate_coverage,
        existing_coverage,
        need_scores,
        opt.num_pharmacies,
        opt.min_distance_miles,
    )

    after_coverage = dict(existing_coverage)
    for cid in selection.selected_ids:
        for geoid, frac in candidate_coverage[cid].items():
            prev = after_coverage.get(geoid, 0.0)
            after_coverage[geoid] = 1.0 - (1.0 - prev) * (1.0 - frac)

    def _with_coverage_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = gdf.copy()
        out["coverage_before_pct"] = out["geoid"].map(existing_coverage).fillna(0.0) * 100.0
        out["coverage_after_pct"] = out["geoid"].map(after_coverage).fillna(0.0) * 100.0
        out["growth_pct"] = (out["coverage_after_pct"] - out["coverage_before_pct"]).clip(lower=0)
        return out.to_crs("EPSG:4326")

    tracts_out = _with_coverage_columns(tracts)
    tract_residential_out = _with_coverage_columns(prepared.tract_residential)

    # Centroid in the projected CRS (meters) -- computing centroids
    # directly in lat/lon distorts them for anything but tiny polygons.
    # Reused for both the community-area join and nearest-pharmacy distance.
    # Joined by geoid (.map()), never by raw positional `.values` -- see
    # the note in get_preview_layer above on why that's not safe in general.
    tract_centroids_3857 = gpd.GeoDataFrame(
        tracts[["geoid"]], geometry=tracts.geometry.centroid, crs=tracts.crs
    )
    nearest_by_geoid = dict(zip(
        tract_centroids_3857["geoid"],
        cov_mod.nearest_distance_miles(tract_centroids_3857, bundle.pharmacies),
    ))
    tracts_out["nearest_pharmacy_miles"] = tracts_out["geoid"].map(nearest_by_geoid)

    # Optional enrichment: Chicago community areas (None elsewhere) and
    # Opportunity Zone designation (any city, since it's a tract attribute).
    community_areas = _get_community_areas(prepared.city_key)
    if community_areas is not None:
        centroids_4326 = tract_centroids_3857.to_crs("EPSG:4326")
        joined = gpd.sjoin(centroids_4326, community_areas[["community_area", "geometry"]], how="left", predicate="within")
        community_by_geoid = dict(zip(joined["geoid"], joined["community_area"]))
        tracts_out["community_area"] = tracts_out["geoid"].map(community_by_geoid)
    else:
        tracts_out["community_area"] = None

    tracts_out["is_opportunity_zone"] = tracts_out["geoid"].isin(oz_tracts)

    # "High-need" = top third of tracts by the need index actually in use
    # (population-weighted, percentile-composite or flat population --
    # see need_index.py) -- a simple, explainable tier rather than an
    # arbitrary cutoff, used only for reporting impact, not for scoring.
    need_percentile = need_index.percentile_rank(tracts_out["weighted_need"])
    tracts_out["is_high_need"] = need_percentile >= (200.0 / 3.0)
    tracts_out = _attach_tooltip_factors(tracts_out, prepared)

    # tract_residential shares the same attribute columns by geoid (its
    # geometry is just clipped to residential+commercial land) -- merge
    # them on so the results map's tooltips have the same richness as the
    # attribute table, without recomputing any of it.
    _attr_cols = [
        "geoid", "community_area", "is_opportunity_zone", "is_high_need", "nearest_pharmacy_miles",
        "no_vehicle_pct", "poverty_pct", "age65_pct", "ambulatory_disability_pct",
        "uninsured_pct", "chronic_burden_pct",
    ]
    tract_residential_out = tract_residential_out.merge(
        tracts_out[[c for c in _attr_cols if c in tracts_out.columns]], on="geoid", how="left"
    )

    all_candidates_out = all_candidates.copy()
    all_candidates_out["selected"] = all_candidates_out["candidate_id"].isin(
        selection.selected_ids
    )
    all_candidates_out["reward"] = all_candidates_out["candidate_id"].map(
        selection.marginal_reward
    )
    nearest_by_candidate_id = dict(zip(
        all_candidates["candidate_id"],
        cov_mod.nearest_distance_miles(all_candidates, bundle.pharmacies),
    ))
    all_candidates_out["nearest_pharmacy_miles"] = all_candidates_out["candidate_id"].map(nearest_by_candidate_id)
    rank_by_id = {cid: i + 1 for i, cid in enumerate(selection.selected_ids)}
    all_candidates_out["rank"] = all_candidates_out["candidate_id"].map(rank_by_id)

    population_all = dict(zip(tracts_out["geoid"], tracts_out["TotalPopulation"].fillna(0.0)))
    pop_reached = _reached_per_site(selection.selected_ids, candidate_coverage, existing_coverage, population_all)
    all_candidates_out["population_reached"] = all_candidates_out["candidate_id"].map(pop_reached)

    high_need_geoids = set(tracts_out.loc[tracts_out["is_high_need"], "geoid"])
    population_high_need = {g: (p if g in high_need_geoids else 0.0) for g, p in population_all.items()}
    high_need_reached = _reached_per_site(
        selection.selected_ids, candidate_coverage, existing_coverage, population_high_need
    )
    all_candidates_out["high_need_population_reached"] = all_candidates_out["candidate_id"].map(high_need_reached)

    # OZ status and the candidate's "home" community area are both
    # properties of where the site itself sits -- use the point-in-tract
    # lookup from the OZ filter step, not buffer overlap.
    oz_by_geoid = dict(zip(tracts_out["geoid"], tracts_out["is_opportunity_zone"]))
    community_by_geoid = dict(zip(tracts_out["geoid"], tracts_out["community_area"]))

    all_candidates_out["is_opportunity_zone"] = all_candidates_out["candidate_id"].map(
        lambda cid: oz_by_geoid.get(candidate_tract.get(cid), False)
    )
    all_candidates_out["community_area"] = all_candidates_out["candidate_id"].map(
        lambda cid: community_by_geoid.get(candidate_tract.get(cid))
    )
    all_candidates_out["site_label"] = all_candidates_out.apply(
        lambda row: f"{row['community_area']} — {row['label']}" if row["community_area"] else row["label"],
        axis=1,
    )

    if community_areas is not None:
        selected_mask = all_candidates_out["candidate_id"].isin(selection.selected_ids)
        selected_3857 = all_candidates_out.loc[selected_mask].to_crs(cov_mod.PROJECTED_CRS)
        neighborhoods_by_id = {}
        for cid, geom_3857 in zip(
            all_candidates_out.loc[selected_mask, "candidate_id"], selected_3857.geometry
        ):
            buf = geom_3857.buffer(opt.desert_radius_miles * cov_mod.METERS_PER_MILE)
            buf_4326 = gpd.GeoSeries([buf], crs=cov_mod.PROJECTED_CRS).to_crs("EPSG:4326").iloc[0]
            touching = community_areas[community_areas.geometry.intersects(buf_4326)]
            neighborhoods_by_id[cid] = ", ".join(sorted(touching["community_area"])) if not touching.empty else ""
        all_candidates_out["neighborhoods"] = all_candidates_out["candidate_id"].map(neighborhoods_by_id)
    else:
        all_candidates_out["neighborhoods"] = None

    all_candidates_out = all_candidates_out.to_crs("EPSG:4326")

    pharmacies_out = bundle.pharmacies.to_crs("EPSG:4326")

    newly_covered_by_tract = (
        (tracts_out["coverage_after_pct"] - tracts_out["coverage_before_pct"])
        / 100.0
        * tracts_out["TotalPopulation"]
    ).clip(lower=0)
    newly_covered_pop = float(newly_covered_by_tract.sum())
    high_need_newly_covered = float(newly_covered_by_tract[tracts_out["is_high_need"]].sum())

    if community_areas is not None:
        impacted_count = int(
            tracts_out.loc[tracts_out["growth_pct"] > 0.5, "community_area"].nunique()
        )
        impact_label = "Communities impacted"
    else:
        impacted_count = int((tracts_out["growth_pct"] > 0.5).sum())
        impact_label = "Tracts improved"

    equity = _compute_equity_summary(tracts_out, prepared.acs_factors)

    summary = {
        "radius_status": radius_status,
        "coverage_status": coverage_status,
        "num_selected": len(selection.selected_ids),
        "num_requested": opt.num_pharmacies,
        "num_candidates_evaluated": len(all_candidates_out),
        "tracts_improved": int((tracts_out["growth_pct"] > 0.5).sum()),
        "impact_label": impact_label,
        "impact_count": impacted_count,
        "newly_covered_population": newly_covered_pop,
        "high_need_newly_covered_population": high_need_newly_covered,
        "avg_coverage_before_pct": float(tracts_out["coverage_before_pct"].mean()),
        "avg_coverage_after_pct": float(tracts_out["coverage_after_pct"].mean()),
        **equity,
    }

    community_impact = _compute_community_impact(tracts_out, all_candidates_out, prepared.acs_factors)

    return PipelineResult(
        tracts=tracts_out,
        tract_residential=tract_residential_out,
        candidates=all_candidates_out,
        pharmacies=pharmacies_out,
        community_impact=community_impact,
        summary=summary,
        center=bundle.center,
    )


def run(
    city_key: str,
    need_weights: NeedWeights,
    opt: OptConfig,
    candidate_cfg: CandidateConfig,
    use_health_priorities: bool = False,
) -> PipelineResult:
    """Convenience wrapper for scripts/tests that don't need the two phases
    split apart (e.g. don't need to show the need map before optimizing)."""
    prepared = prepare_city(city_key, need_weights, use_health_priorities=use_health_priorities)
    return run_optimization(prepared, opt, candidate_cfg)


def run_sensitivity_analysis(
    prepared: PreparedCity,
    opt: OptConfig,
    candidate_cfg: CandidateConfig,
    n_random_samples: int = 20,
    seed: int = 42,
) -> dict:
    """Test how stable the siting recommendations are across different weight assumptions.

    Generates candidates and computes coverage fractions once (the expensive
    part), then runs the greedy optimizer across n_random_samples random
    Dirichlet weight samples plus the named strategy presets.  Returns which
    candidate sites appear in what fraction of all runs — "stability."

    A site with stability ≥ 0.8 is recommended by 80%+ of all weight
    combinations, making it robust to disagreements about how to weight the
    need factors.

    Returns a dict:
      candidates    GeoDataFrame (EPSG:4326): candidate_id, label, stability,
                    geometry.  Only sites with stability > 0 are meaningful.
      total_runs    int — total optimizer runs performed.
      n_random      int — number of random Dirichlet samples used.
      high_stability_count  int — sites with stability ≥ 0.8.
      strategy_names        list[str] — named presets also included.
    """
    from .config import DEFAULT_WEIGHTS, STRATEGY_PRESETS

    bundle = prepared.bundle
    tracts = prepared.tracts

    # Compute coverage once — the most expensive step (unchanged across runs)
    cov_index, _ = _resolve_coverage_index(prepared.city_key, prepared, opt)
    radius, _ = _resolve_desert_radius(prepared.city_key, tracts, opt)
    existing_coverage = cov_mod.coverage_from_points(bundle.pharmacies, cov_index, radius)
    oz_tracts = _get_opportunity_zone_tracts(prepared.city_key, tracts["geoid"].tolist())

    all_candidates = cand_mod.generate_grid_candidates(
        bundle.zoned_land, candidate_cfg.grid_spacing_meters
    )
    all_candidates = cand_mod.exclude_near_existing(
        all_candidates, bundle.pharmacies, opt.min_distance_miles
    )
    if opt.restrict_to_opportunity_zones:
        candidate_tract = _assign_point_in_tract(all_candidates, tracts)
        all_candidates = _filter_to_opportunity_zones(all_candidates, candidate_tract, oz_tracts)

    candidate_coverage = {
        cid: cov_index.coverage_fractions(geom, radius)
        for cid, geom in zip(all_candidates["candidate_id"], all_candidates.geometry)
    }

    # Weight profiles: random Dirichlet samples + named presets
    factor_keys = list(DEFAULT_WEIGHTS.keys())
    rng = np.random.default_rng(seed)
    random_weight_list = [
        dict(zip(factor_keys, w))
        for w in rng.dirichlet(np.ones(len(factor_keys)), size=n_random_samples)
    ]
    preset_items = [(name, w) for name, w in STRATEGY_PRESETS.items() if w is not None]

    selection_counts: dict = {cid: 0 for cid in candidate_coverage}
    total_runs = 0

    for weights_dict in random_weight_list:
        need_df = need_index.compute_need_scores(
            bundle.tracts, bundle.places_scores, prepared.acs_factors,
            weights_dict, use_health_priorities=True,
        )
        need_scores = dict(zip(need_df["geoid"], need_df["weighted_need"].fillna(0.0)))
        sel = optimize.greedy_select(
            all_candidates, candidate_coverage, existing_coverage,
            need_scores, opt.num_pharmacies, opt.min_distance_miles,
        )
        for cid in sel.selected_ids:
            selection_counts[cid] += 1
        total_runs += 1

    for _name, weights_dict in preset_items:
        need_df = need_index.compute_need_scores(
            bundle.tracts, bundle.places_scores, prepared.acs_factors,
            weights_dict, use_health_priorities=True,
        )
        need_scores = dict(zip(need_df["geoid"], need_df["weighted_need"].fillna(0.0)))
        sel = optimize.greedy_select(
            all_candidates, candidate_coverage, existing_coverage,
            need_scores, opt.num_pharmacies, opt.min_distance_miles,
        )
        for cid in sel.selected_ids:
            selection_counts[cid] += 1
        total_runs += 1

    stability = {cid: count / total_runs for cid, count in selection_counts.items()}

    out = all_candidates.copy()
    out["stability"] = out["candidate_id"].map(stability).fillna(0.0)
    out = out.to_crs("EPSG:4326")

    return {
        "candidates": out,
        "total_runs": total_runs,
        "n_random": n_random_samples,
        "high_stability_count": int((out["stability"] >= 0.8).sum()),
        "strategy_names": [name for name, _ in preset_items],
    }
