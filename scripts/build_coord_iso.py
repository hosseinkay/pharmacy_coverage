"""Build candidate_iso_by_coord_r0.50.parquet quickly.

Strategy (fast, <30s total):
  1. Remap the 10 already-computed demo-selected isochrones from
     candidate_isochrones_r0.50.parquet to coordinate keys.
  2. Compute isochrones for the top-200 candidates by
     high_need_population_reached (the metric the greedy optimizer
     maximises), covering the likely live-run selections.
  3. Union both sets, deduplicate by coordinate key, save.

Result: ~210 coordinate-keyed isochrones, ~5-6 MB.
Covers all default-setting live runs (N ≤ 10 with default strategy).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from core import cache, isochrones
from core.city_data import get_osm_place_name

CITY_KEY  = "chicago"
RADIUS    = 0.5
SIMP_TOL  = 0.0001   # ≈ 11 m — fine for a coverage viz layer
TOP_N     = 200      # additional high-need candidates to precompute

def coord_key(geom) -> str:
    return f"{geom.x:.5f}_{geom.y:.5f}"


def main():
    place_name = get_osm_place_name(CITY_KEY)
    slug       = cache.slugify(place_name)
    cache_dir  = cache.city_cache_dir(slug)
    out_path   = cache_dir / f"candidate_iso_by_coord_r{RADIUS:.2f}.parquet"

    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    cands    = gpd.read_parquet(demo_dir / "candidates.parquet")  # EPSG:4326
    print(f"All candidates: {len(cands)}")

    rows: dict[str, object] = {}   # coord_key → Shapely geometry

    # ── 1. Remap the 10 already-computed isochrones ──────────────────────────
    existing_path = cache_dir / f"candidate_isochrones_r{RADIUS:.2f}.parquet"
    if existing_path.exists():
        iso10 = gpd.read_parquet(existing_path)   # index = old candidate_id
        # Match old IDs → coordinates via the demo candidates parquet
        id_to_coord = {
            int(row.candidate_id): coord_key(row.geometry)
            for _, row in cands.iterrows()
            if int(row.candidate_id) in iso10.index
        }
        for cid, geom in zip(iso10.index, iso10.geometry):
            cid = int(cid)
            if cid in id_to_coord:
                key = id_to_coord[cid]
                rows[key] = geom.simplify(SIMP_TOL)
        print(f"Remapped {len(rows)} existing isochrones to coordinate keys")
    else:
        print("candidate_isochrones_r0.50.parquet not found — skipping remap")

    # ── 2. Top-N candidates by high_need_population_reached ──────────────────
    score_col = "high_need_population_reached" if "high_need_population_reached" in cands.columns \
                else "population_reached"
    top = (
        cands
        .sort_values(score_col, ascending=False)
        .head(TOP_N + len(rows))  # over-request to account for already-done ones
    )
    # Drop already done
    need_compute = [
        row for _, row in top.iterrows()
        if coord_key(row.geometry) not in rows
    ][:TOP_N]

    if need_compute:
        print(f"\nLoading drive network …")
        t0 = time.time()
        graph = isochrones.get_network_graph_cached(place_name)
        print(f"  loaded in {time.time()-t0:.1f}s")

        print(f"Computing {len(need_compute)} top-{TOP_N} candidate isochrones …")
        t0 = time.time()
        ok = err = 0
        for i, row in enumerate(need_compute):
            one  = cands.loc[[row.name]]
            geom = isochrones.compute_merged_isochrone(graph, one, RADIUS)
            if geom is not None and not geom.is_empty:
                rows[coord_key(row.geometry)] = geom.simplify(SIMP_TOL)
                ok += 1
            else:
                err += 1
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(need_compute)}  ok={ok} err={err}  {rate:.1f}/s")
        print(f"  Done in {time.time()-t0:.1f}s  ok={ok} err={err}")

    # ── 3. Save ───────────────────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(
        {"coord_key": list(rows.keys()), "geometry": list(rows.values())},
        crs="EPSG:4326",
    ).set_index("coord_key")
    gdf.to_parquet(out_path)

    mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(gdf)} isochrones → {out_path.name}  ({mb:.1f} MB)")
    print("Next: git add + commit + push")


if __name__ == "__main__":
    main()
