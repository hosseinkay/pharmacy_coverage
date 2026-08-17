"""Fast parallel precompute of isochrones for all 7283 candidates.

Uses 2 worker processes (each loads its own graph copy, ~800 MB each).
On 16 GB RAM this is safe.  Saves incrementally every 300 candidates so
partial results can be committed while the rest runs.

Final output: .cache/chicago-illinois-usa/candidate_iso_by_coord_r0.50.parquet
Index: coord_key "lon_lat" (5-decimal EPSG:4326)
"""
from __future__ import annotations

import sys, time, pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from shapely.ops import unary_union

from core import cache, isochrones
from core.city_data import get_osm_place_name

CITY_KEY  = "chicago"
RADIUS    = 0.5
SIMP_TOL  = 0.0001
N_WORKERS = 2          # each loads its own ~800 MB graph
BATCH_SIZE = 300       # save parquet after every BATCH_SIZE candidates

# ── worker init ─────────────────────────────────────────────────────────────

_graph = None  # per-worker global


def _init_worker(place_name: str):
    global _graph
    _graph = isochrones.get_network_graph_cached(place_name)


def _compute_one(args):
    """args = (coord_key, x, y)  — uses the per-process _graph."""
    import osmnx as ox
    import networkx as nx
    coord_key, x, y = args
    try:
        node = ox.distance.nearest_nodes(_graph, x, y)
        radius_m = RADIUS * 1609.344
        sub = nx.ego_graph(_graph, node, radius=radius_m, distance="length")
        if sub.number_of_edges() == 0:
            return coord_key, None
        edges = ox.graph_to_gdfs(sub, nodes=False, edges=True)
        if edges.empty:
            return coord_key, None
        merged = edges.to_crs("EPSG:3857").geometry.buffer(40).union_all()
        poly4326 = gpd.GeoSeries([merged], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        return coord_key, poly4326.simplify(SIMP_TOL)
    except Exception:
        return coord_key, None


# ── main ────────────────────────────────────────────────────────────────────

def main():
    place_name = get_osm_place_name(CITY_KEY)
    slug       = cache.slugify(place_name)
    cache_dir  = cache.city_cache_dir(slug)
    out_path   = cache_dir / f"candidate_iso_by_coord_r{RADIUS:.2f}.parquet"

    # Load all candidates
    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    cands    = gpd.read_parquet(demo_dir / "candidates.parquet")   # EPSG:4326

    # Load existing results to skip already-done ones
    done: dict[str, object] = {}
    if out_path.exists():
        existing = gpd.read_parquet(out_path)
        done = dict(zip(existing.index, existing.geometry))
        print(f"Loaded {len(done)} already-computed isochrones")

    # Build work list: skip already-done coord keys
    todo = []
    for _, row in cands.iterrows():
        key = f"{row.geometry.x:.5f}_{row.geometry.y:.5f}"
        if key not in done:
            todo.append((key, row.geometry.x, row.geometry.y))

    if not todo:
        print("All candidates already computed!")
        return

    total = len(todo)
    print(f"Need to compute: {total}  (skipping {len(done)} already done)")
    print(f"Using {N_WORKERS} workers — each loads ~800 MB graph copy")
    print(f"Saving every {BATCH_SIZE} results  ETA ~{total * 0.18 / N_WORKERS / 60:.0f} min\n")

    t_start = time.time()
    results: dict[str, object] = dict(done)  # accumulate all results
    batch_done = 0

    with Pool(N_WORKERS, initializer=_init_worker, initargs=(place_name,)) as pool:
        for i, (key, geom) in enumerate(pool.imap_unordered(_compute_one, todo, chunksize=10)):
            if geom is not None and not geom.is_empty:
                results[key] = geom
            batch_done += 1

            if batch_done % 50 == 0:
                elapsed  = time.time() - t_start
                rate     = batch_done / elapsed
                eta_min  = (total - batch_done) / (rate * N_WORKERS) / 60 if rate else 0
                pct      = batch_done / total * 100
                print(f"  {batch_done:5d}/{total} ({pct:.0f}%)  "
                      f"ok={len(results)-len(done)}  "
                      f"rate={rate*N_WORKERS:.1f}/s  ETA~{eta_min:.0f}min")

            # Incremental save every BATCH_SIZE
            if batch_done % BATCH_SIZE == 0:
                _save(results, out_path)
                print(f"  ↳ Saved {len(results)} isochrones to {out_path.name}")

    # Final save
    _save(results, out_path)
    mb      = out_path.stat().st_size / 1024 / 1024
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Saved {len(results)} isochrones → {out_path.name}  ({mb:.1f} MB)")


def _save(results: dict, path: Path):
    rows = [{"coord_key": k, "geometry": v} for k, v in results.items()]
    gdf  = gpd.GeoDataFrame(rows, crs="EPSG:4326").set_index("coord_key")
    gdf.to_parquet(path)


if __name__ == "__main__":
    main()
