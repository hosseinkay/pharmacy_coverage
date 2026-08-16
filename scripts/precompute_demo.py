#!/usr/bin/env python
"""Precompute and save the Chicago demo snapshot.

Run this script once locally (with a warm cache / Census API key) before
deploying. The outputs are committed to data/demo/ and loaded instantly at
app startup so cold visitors never wait for external APIs or heavy computation.

Usage
-----
From the pharmacy-desert-app/ directory:

    python scripts/precompute_demo.py

Or with explicit settings:

    python scripts/precompute_demo.py --pharmacies 10 --radius 0.5

The script uses the live pipeline with default "Balanced need" weights and
the selected settings, then serialises the results. Commit data/demo/ to git.

Expected runtime: 30–120 s on first run (OSM + ACS fetches), ~20 s on
a warm cache.

Expected output size: 2–6 MB total (well within GitHub limits).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sure core/ is importable regardless of where the script is invoked from
_repo = Path(__file__).resolve().parent.parent  # = pharmacy-desert-app/
sys.path.insert(0, str(_repo))  # makes `from core import …` work

from core import pipeline
from core.config import CandidateConfig, NeedWeights, OptConfig, DEFAULT_WEIGHTS
from core.demo_data import save_demo_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Chicago demo snapshot.")
    parser.add_argument("--pharmacies", type=int, default=10,
                        help="Number of pharmacies to site (default: 10)")
    parser.add_argument("--radius", type=float, default=0.5,
                        help="Desert radius in miles (default: 0.5 = walking)")
    parser.add_argument("--grid-spacing", type=float, default=200.0,
                        help="Candidate grid spacing in metres (default: 200)")
    args = parser.parse_args()

    print("=" * 60)
    print("Chicago Pharmacy Desert Planner — demo precompute")
    print("=" * 60)
    print(f"  Pharmacies  : {args.pharmacies}")
    print(f"  Radius      : {args.radius} mi")
    print(f"  Grid spacing: {args.grid_spacing} m")
    print()

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Prepare the city (loads data, computes need index, clips geometry)
    # ------------------------------------------------------------------
    print("Step 1/3  Preparing city data…")
    need_weights = NeedWeights(weights=dict(DEFAULT_WEIGHTS))
    prepared = pipeline.prepare_city(
        "chicago", need_weights, use_health_priorities=True
    )
    print(f"          Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Run optimisation with default settings
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    print("Step 2/3  Running optimisation…")
    opt = OptConfig(
        num_pharmacies=args.pharmacies,
        min_distance_miles=args.radius,
        desert_radius_miles=args.radius,
        use_tiered_radius=False,
        extended_radius_miles=args.radius,
        use_population_weighted_coverage=False,
        restrict_to_opportunity_zones=False,
    )
    result = pipeline.run_optimization(
        prepared, opt, CandidateConfig(grid_spacing_meters=args.grid_spacing)
    )
    n_sel = result.summary["num_selected"]
    n_cov = int(result.summary["newly_covered_population"])
    print(f"          Selected {n_sel} sites | ~{n_cov:,} residents newly covered")
    print(f"          Done in {time.perf_counter() - t1:.1f}s")

    # ------------------------------------------------------------------
    # 3. Compute preview layer (need map choropleth)
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    print("Step 3/3  Computing preview layer…")
    preview_layer = pipeline.get_preview_layer(prepared, args.radius)
    print(f"          Done in {time.perf_counter() - t2:.1f}s")

    # ------------------------------------------------------------------
    # 4. Save snapshot
    # ------------------------------------------------------------------
    meta = {
        "num_pharmacies": args.pharmacies,
        "base_radius": args.radius,
        "strategy": "Balanced need (recommended)",
        "grid_spacing_meters": args.grid_spacing,
    }
    save_demo_snapshot(preview_layer, result, meta)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.0f}s")
    print()
    print("Next steps:")
    print("  git add pharmacy-desert-app/data/demo/")
    print("  git commit -m 'chore: add precomputed Chicago demo snapshot'")
    print("  git push")


if __name__ == "__main__":
    main()
