"""Pre-compute street-network isochrones for ALL demo candidates.

Writes .cache/chicago-illinois-usa/candidate_iso_by_coord_r0.50.parquet

Index: coordinate key  "lon_lat"  (5-decimal EPSG:4326, ≈ 1 m precision)
Geometry: EPSG:4326 simplified road-corridor polygon per candidate

The coordinate key is stable across pipeline runs:
  - generate_grid_candidates(zoned_land) produces the same EPSG:3857 grid
    every time (deterministic from committed zoned_land.parquet + numpy)
  - EPSG:3857 → EPSG:4326 is exact for the same input
  → same geographic point → same coordinate key regardless of candidate_id

This lets get_new_coverage_shape look up by coordinate instead of ID,
solving the ID-mismatch problem where the demo uses sparse IDs (0-14330)
and live runs use sequential IDs (0-7282).

Run once locally; commit the parquet (~15-17 MB).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from core import cache, isochrones
from core.city_data import get_osm_place_name

CITY_KEY = "chicago"
RADIUS = 0.5
SIMPLIFY_TOL = 0.0001  # degrees ≈ 11 m — good enough for a coverage viz


def main():
    # ── Load all demo candidates (7283 rows, EPSG:4326) ─────────────────────
    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    cands = gpd.read_parquet(demo_dir / "candidates.parquet")
    print(f"Candidates: {len(cands)}  |  CRS: {cands.crs}")

    # ── Load network graph ───────────────────────────────────────────────────
    place_name = get_osm_place_name(CITY_KEY)
    slug = cache.slugify(place_name)
    cache_dir = cache.city_cache_dir(slug)
    out_path = cache_dir / f"candidate_iso_by_coord_r{RADIUS:.2f}.parquet"

    print(f"Loading drive network from {cache_dir / 'drive_network.graphml'} …")
    t_load = time.time()
    graph = isochrones.get_network_graph_cached(place_name)
    print(f"  loaded in {time.time() - t_load:.1f}s\n")

    # ── Compute one isochrone per candidate ──────────────────────────────────
    total = len(cands)
    rows = []
    errors = 0
    t0 = time.time()

    for i, (_, row) in enumerate(cands.iterrows()):
        one = cands.loc[[_]]
        geom = isochrones.compute_merged_isochrone(graph, one, RADIUS)

        if geom is None or geom.is_empty:
            errors += 1
            continue

        # Simplify to reduce file size (11 m precision — fine for a viz layer)
        geom_s = geom.simplify(SIMPLIFY_TOL)
        key = f"{row.geometry.x:.5f}_{row.geometry.y:.5f}"
        rows.append({"coord_key": key, "geometry": geom_s})

        # Progress every 50
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i+1:5d}/{total}  "
                f"rows={len(rows):5d}  errors={errors}  "
                f"rate={rate:.1f}/s  ETA={eta/60:.1f} min"
            )

    # ── Save ─────────────────────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").set_index("coord_key")
    gdf.to_parquet(out_path)

    mb = out_path.stat().st_size / 1024 / 1024
    total_t = time.time() - t0
    print(f"\nSaved: {out_path.name}  {mb:.1f} MB  ({total_t/60:.1f} min total)")
    print(f"Rows: {len(gdf)}  |  Errors/skipped: {errors}")
    print(
        "\nNext: add  !.cache/chicago-illinois-usa/candidate_iso_by_coord_r0.50.parquet"
        "  to .gitignore, then commit and push."
    )


if __name__ == "__main__":
    main()
