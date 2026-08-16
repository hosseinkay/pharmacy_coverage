"""Tests for the demo snapshot serialization / deserialization round-trip.

The demo data module is the gatekeeper for instant cold-start performance on
Streamlit Community Cloud. These tests verify:
  1. save_demo_snapshot() writes all required files.
  2. load_demo_snapshot() reconstructs objects that match the originals.
  3. load_demo_snapshot() returns None gracefully when data is absent or corrupt.
  4. _clean_for_json() converts every numpy scalar type to a JSON-native type.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

# We monkey-patch DEMO_DIR before importing the module, so tests never touch
# the real data/demo/ directory.
_FAKE_DEMO_DIR = Path(tempfile.mkdtemp()) / "demo"


@pytest.fixture(autouse=True)
def patch_demo_dir(tmp_path):
    """Re-direct all demo I/O to a fresh temp directory for every test."""
    import core.demo_data as dd
    original = dd.DEMO_DIR
    dd.DEMO_DIR = tmp_path / "demo"
    yield tmp_path / "demo"
    dd.DEMO_DIR = original


# ---------------------------------------------------------------------------
# Helpers — minimal fake pipeline objects
# ---------------------------------------------------------------------------

UNIT = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def _fake_gdf(crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        {"geoid": ["T1", "T2"], "value": [1.0, 2.0]},
        geometry=[UNIT, UNIT],
        crs=crs,
    )


def _fake_result():
    from core.pipeline import PipelineResult

    candidates = gpd.GeoDataFrame(
        {"candidate_id": [1, 2], "selected": [True, False]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    community_impact = pd.DataFrame({"community": ["A"], "pop_reached": [1000]})

    return PipelineResult(
        tracts=_fake_gdf(),
        tract_residential=_fake_gdf(),
        candidates=candidates,
        pharmacies=gpd.GeoDataFrame(
            {"name": ["P1"]}, geometry=[Point(0, 0)], crs="EPSG:4326"
        ),
        community_impact=community_impact,
        summary={
            "num_selected": np.int64(2),
            "newly_covered_population": np.float64(5000.1),
            "pct_gain": np.float32(12.5),
            "nan_value": float("nan"),
        },
        center=(41.85, -87.65),
    )


def _fake_meta(base_radius: float = 0.5):
    return {
        "num_pharmacies": 10,
        "base_radius": base_radius,
        "strategy": "Balanced need (recommended)",
    }


# ---------------------------------------------------------------------------
# save + load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_all_required_files(patch_demo_dir):
    from core.demo_data import save_demo_snapshot

    save_demo_snapshot(_fake_gdf(), _fake_result(), _fake_meta())

    required = [
        "preview_layer.parquet",
        "tract_residential.parquet",
        "candidates.parquet",
        "pharmacies.parquet",
        "tracts.parquet",
        "community_impact.parquet",
        "summary.json",
        "meta.json",
    ]
    for fname in required:
        assert (patch_demo_dir / fname).exists(), f"Missing: {fname}"


def test_load_returns_none_when_directory_absent(patch_demo_dir):
    from core.demo_data import load_demo_snapshot

    # patch_demo_dir doesn't exist yet (no save called)
    assert load_demo_snapshot() is None


def test_load_returns_none_when_a_required_file_is_missing(patch_demo_dir):
    from core.demo_data import load_demo_snapshot, save_demo_snapshot

    save_demo_snapshot(_fake_gdf(), _fake_result(), _fake_meta())
    (patch_demo_dir / "summary.json").unlink()  # remove one required file

    assert load_demo_snapshot() is None


def test_load_reconstructs_center_tuple(patch_demo_dir):
    from core.demo_data import load_demo_snapshot, save_demo_snapshot

    save_demo_snapshot(_fake_gdf(), _fake_result(), _fake_meta())
    snap = load_demo_snapshot()

    assert snap is not None
    assert snap.result.center == (41.85, -87.65)


def test_load_reconstructs_summary_values(patch_demo_dir):
    from core.demo_data import load_demo_snapshot, save_demo_snapshot

    save_demo_snapshot(_fake_gdf(), _fake_result(), _fake_meta())
    snap = load_demo_snapshot()

    assert snap.result.summary["num_selected"] == 2
    assert snap.result.summary["newly_covered_population"] == pytest.approx(5000.1)
    assert snap.result.summary["nan_value"] is None  # NaN → null in JSON → None


def test_load_reconstructs_meta(patch_demo_dir):
    from core.demo_data import load_demo_snapshot, save_demo_snapshot

    save_demo_snapshot(_fake_gdf(), _fake_result(), _fake_meta(base_radius=1.0))
    snap = load_demo_snapshot()

    assert snap.meta["base_radius"] == 1.0
    assert snap.meta["strategy"] == "Balanced need (recommended)"


def test_load_reconstructs_geodataframes_with_geometry(patch_demo_dir):
    from core.demo_data import load_demo_snapshot, save_demo_snapshot

    save_demo_snapshot(_fake_gdf(), _fake_result(), _fake_meta())
    snap = load_demo_snapshot()

    assert snap is not None
    assert "geometry" in snap.result.tracts.columns
    assert len(snap.result.tracts) == 2
    assert len(snap.result.candidates) == 2


def test_load_returns_none_on_corrupt_parquet(patch_demo_dir):
    from core.demo_data import load_demo_snapshot, save_demo_snapshot

    save_demo_snapshot(_fake_gdf(), _fake_result(), _fake_meta())
    # Corrupt one parquet file
    (patch_demo_dir / "tracts.parquet").write_bytes(b"not-a-parquet")

    # Should warn and return None, not raise
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = load_demo_snapshot()
    assert result is None
    assert any("Demo snapshot failed to load" in str(warning.message) for warning in w)


def test_community_impact_optional(patch_demo_dir):
    from core.demo_data import load_demo_snapshot, save_demo_snapshot
    from core.pipeline import PipelineResult

    result = _fake_result()
    # Build a result with no community_impact
    result_no_ci = PipelineResult(
        tracts=result.tracts,
        tract_residential=result.tract_residential,
        candidates=result.candidates,
        pharmacies=result.pharmacies,
        community_impact=None,
        summary=result.summary,
        center=result.center,
    )
    save_demo_snapshot(_fake_gdf(), result_no_ci, _fake_meta())
    snap = load_demo_snapshot()

    assert snap is not None
    assert snap.result.community_impact is None


# ---------------------------------------------------------------------------
# _clean_for_json — numpy scalar handling
# ---------------------------------------------------------------------------


def test_clean_for_json_converts_numpy_int():
    from core.demo_data import _clean_for_json

    assert isinstance(_clean_for_json(np.int64(42)), int)
    assert _clean_for_json(np.int64(42)) == 42


def test_clean_for_json_converts_numpy_float():
    from core.demo_data import _clean_for_json

    assert isinstance(_clean_for_json(np.float32(3.14)), float)


def test_clean_for_json_converts_numpy_bool():
    from core.demo_data import _clean_for_json

    assert _clean_for_json(np.bool_(True)) is True
    assert isinstance(_clean_for_json(np.bool_(True)), bool)


def test_clean_for_json_nan_becomes_none():
    from core.demo_data import _clean_for_json

    assert _clean_for_json(float("nan")) is None
    assert _clean_for_json(np.float64("nan")) is None


def test_clean_for_json_nested_dict_and_list():
    from core.demo_data import _clean_for_json

    raw = {"a": np.int64(1), "b": [np.float32(2.0), {"c": np.bool_(False)}]}
    cleaned = _clean_for_json(raw)

    assert json.dumps(cleaned)  # must be serializable — raises if not
    assert cleaned["a"] == 1
    assert cleaned["b"][0] == pytest.approx(2.0)
    assert cleaned["b"][1]["c"] is False
