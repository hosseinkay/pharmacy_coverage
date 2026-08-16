"""Serialization / deserialization of pipeline outputs for the public demo.

The deployed application should never force a cold visitor to wait minutes for
external API calls or expensive spatial operations. This module provides:

  save_demo_snapshot()  -- run once locally, writes data/demo/
  load_demo_snapshot()  -- called at app startup; returns objects or None

When demo data is present the bootstrap section of streamlit_app.py skips
the live pipeline entirely -- the page opens with results immediately. If the
demo directory is missing (developer environment without precomputed data) the
app falls back gracefully to the live pipeline, preserving the full rebuild
workflow.

Design constraint: this module must not import Streamlit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from .pipeline import PipelineResult

# All demo files live under pharmacy-desert-app/data/demo/
DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DemoSnapshot:
    """Everything the app needs to render the initial experience instantly."""
    preview_layer: gpd.GeoDataFrame     # need map choropleth + pharmacy dots
    result: PipelineResult              # default optimization output
    meta: dict                          # num_pharmacies, base_radius, strategy, etc.


def save_demo_snapshot(
    preview_layer: gpd.GeoDataFrame,
    result: PipelineResult,
    meta: dict,
) -> None:
    """Serialize pipeline outputs to data/demo/. Run this script locally then
    commit the directory. Files are small (~2–5 MB total)."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    preview_layer.to_crs("EPSG:4326").to_parquet(DEMO_DIR / "preview_layer.parquet")
    result.tract_residential.to_crs("EPSG:4326").to_parquet(DEMO_DIR / "tract_residential.parquet")
    result.candidates.to_crs("EPSG:4326").to_parquet(DEMO_DIR / "candidates.parquet")
    result.pharmacies.to_crs("EPSG:4326").to_parquet(DEMO_DIR / "pharmacies.parquet")
    result.tracts.to_crs("EPSG:4326").to_parquet(DEMO_DIR / "tracts.parquet")

    if result.community_impact is not None:
        result.community_impact.to_parquet(DEMO_DIR / "community_impact.parquet")

    # summary dict — convert any numpy scalars that aren't JSON-serializable
    summary_clean = _clean_for_json(result.summary)
    (DEMO_DIR / "summary.json").write_text(
        json.dumps(summary_clean, indent=2), encoding="utf-8"
    )

    meta_out = {
        "center": list(result.center),
        **{k: v for k, v in meta.items()},
    }
    (DEMO_DIR / "meta.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )

    print(f"Demo snapshot saved to {DEMO_DIR}")
    for p in sorted(DEMO_DIR.iterdir()):
        kb = p.stat().st_size / 1024
        print(f"  {p.name:40s}  {kb:7.1f} KB")


def load_demo_snapshot() -> Optional[DemoSnapshot]:
    """Load precomputed demo data. Returns None if data/demo/ is absent,
    allowing the app to fall back to the live pipeline."""
    required = [
        "preview_layer.parquet",
        "tract_residential.parquet",
        "candidates.parquet",
        "pharmacies.parquet",
        "tracts.parquet",
        "summary.json",
        "meta.json",
    ]
    if not DEMO_DIR.exists() or not all((DEMO_DIR / f).exists() for f in required):
        return None

    try:
        preview_layer = gpd.read_parquet(DEMO_DIR / "preview_layer.parquet")
        tract_residential = gpd.read_parquet(DEMO_DIR / "tract_residential.parquet")
        candidates = gpd.read_parquet(DEMO_DIR / "candidates.parquet")
        pharmacies = gpd.read_parquet(DEMO_DIR / "pharmacies.parquet")
        tracts = gpd.read_parquet(DEMO_DIR / "tracts.parquet")

        community_impact: Optional[pd.DataFrame] = None
        if (DEMO_DIR / "community_impact.parquet").exists():
            community_impact = pd.read_parquet(DEMO_DIR / "community_impact.parquet")

        summary = json.loads((DEMO_DIR / "summary.json").read_text(encoding="utf-8"))
        meta = json.loads((DEMO_DIR / "meta.json").read_text(encoding="utf-8"))

        center = tuple(meta["center"])

        result = PipelineResult(
            tracts=tracts,
            tract_residential=tract_residential,
            candidates=candidates,
            pharmacies=pharmacies,
            community_impact=community_impact,
            summary=summary,
            center=center,  # type: ignore[arg-type]
        )

        return DemoSnapshot(
            preview_layer=preview_layer,
            result=result,
            meta=meta,
        )
    except Exception as exc:  # noqa: BLE001
        # Demo data is corrupt or from a different version — fall through to
        # live pipeline rather than crashing the deployed app.
        import warnings
        warnings.warn(
            f"Demo snapshot failed to load ({exc}); falling back to live pipeline.",
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_for_json(obj):
    """Recursively convert numpy scalars and other non-JSON types to natives."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if f != f else f   # NaN (any float NaN) → JSON null
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (obj != obj):  # plain Python NaN
        return None
    return obj
