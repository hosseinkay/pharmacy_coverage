"""Pre-compute per-N merged isochrones for the demo candidates.

Writes one parquet per N (1..10) to .cache/chicago-illinois-usa/:
  new_sites_iso_n01_r0.50.parquet  <- top-1 candidate merged isochrone
  new_sites_iso_n02_r0.50.parquet  <- top-2 candidates merged isochrone
  ...
  new_sites_iso_n10_r0.50.parquet  <- all 10 demo candidates merged isochrone

These mirror exactly how existing_isochrone_r0.50.parquet works for the
green layer (single merged polygon, read with gdf.geometry.iloc[0]).

Run once locally, then commit all 10 parquets.  They are tiny (< 50 KB each).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from core import cache, isochrones
from core.city_data import get_osm_place_name

CITY_KEY = "chicago"
RADIUS = 0.5

# --------------------------------------------------------------------------
# 1. Load demo candidates, sorted by greedy rank (ascending = best first)
# --------------------------------------------------------------------------
demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
cands = gpd.read_parquet(demo_dir / "candidates.parquet")
selected = cands[cands["selected"]].sort_values("rank").reset_index(drop=True)

print(f"Demo selected candidates: {len(selected)}")
print(f"Greedy rank order:")
for _, row in selected.iterrows():
    print(f"  rank={int(row['rank']):2d}  candidate_id={int(row['candidate_id']):6d}  "
          f"({row.geometry.y:.4f}, {row.geometry.x:.4f})")

# --------------------------------------------------------------------------
# 2. Load drive network (must already be cached locally)
# --------------------------------------------------------------------------
place_name = get_osm_place_name(CITY_KEY)
slug = cache.slugify(place_name)
cache_dir = cache.city_cache_dir(slug)

print(f"\nLoading drive network from {cache_dir / 'drive_network.graphml'} …")
t0 = time.time()
graph = isochrones.get_network_graph_cached(place_name)
print(f"  loaded in {time.time() - t0:.1f}s")

# --------------------------------------------------------------------------
# 3. Pre-compute individual per-candidate isochrones (reuse if already done)
# --------------------------------------------------------------------------
print("\nComputing per-candidate isochrones …")
per_candidate: list = []  # list of Shapely polygons, ordered by rank
for i, row in selected.iterrows():
    one = selected.iloc[[i]]  # single-row GDF
    geom = isochrones.compute_merged_isochrone(graph, one, RADIUS)
    per_candidate.append(geom)
    print(f"  rank={int(row['rank']):2d}  {geom.geom_type if geom else 'None'}")

# --------------------------------------------------------------------------
# 4. Write one merged parquet per N
# --------------------------------------------------------------------------
print(f"\nWriting per-N parquets to {cache_dir} …")
total_t = time.time()
for n in range(1, len(selected) + 1):
    polys = [p for p in per_candidate[:n] if p is not None]
    if not polys:
        print(f"  N={n:02d}  no geometry, skipping")
        continue

    merged = gpd.GeoSeries(polys, crs="EPSG:4326").union_all()
    gdf = gpd.GeoDataFrame({"geometry": [merged]}, crs="EPSG:4326")

    out_path = cache_dir / f"new_sites_iso_n{n:02d}_r{RADIUS:.2f}.parquet"
    gdf.to_parquet(out_path)
    kb = out_path.stat().st_size // 1024
    print(f"  N={n:02d}  {merged.geom_type:20s}  {kb:4d} KB  → {out_path.name}")

print(f"\nDone in {time.time() - total_t:.1f}s total")
print("\nNext: add !.cache/chicago-illinois-usa/new_sites_iso_n*_r0.50.parquet to .gitignore")
print("      then commit and push.")
