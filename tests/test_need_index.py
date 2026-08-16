"""Tests for the population-weighted composite need index: each factor is
percentile-ranked within the city before being combined, then the
composite is scaled by population (not the other way around -- the
original model's bug was dividing by population before normalizing).
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from core import need_index

UNIT_SQUARE = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def make_tracts(populations=(1000, 2000, 3000)):
    return gpd.GeoDataFrame(
        {
            "geoid": [f"T{i}" for i in range(len(populations))],
            "TotalPopulation": list(populations),
            "geometry": [UNIT_SQUARE] * len(populations),
        },
        crs="EPSG:4326",
    )


def make_places(diabetes_rates, geoids):
    return pd.DataFrame(
        {
            "geoid": geoids,
            "MeasureId": ["DIABETES"] * len(geoids),
            "Data_Value": diabetes_rates,
        }
    )


def test_population_only_mode_ignores_all_factors():
    tracts = make_tracts((1000, 2000))
    places = make_places([90.0, 5.0], ["T0", "T1"])
    out = need_index.compute_need_scores(
        tracts, places, None, {"chronic_burden": 1.0}, use_health_priorities=False
    ).set_index("geoid")
    assert out.loc["T0", "raw_need_score"] == out.loc["T1", "raw_need_score"] == 1.0
    assert out.loc["T1", "weighted_need"] == pytest.approx(2 * out.loc["T0", "weighted_need"])


def test_higher_chronic_burden_scores_higher_percentile():
    tracts = make_tracts((1000, 1000, 1000))
    places = make_places([90.0, 50.0, 5.0], ["T0", "T1", "T2"])
    out = need_index.compute_need_scores(
        tracts, places, None, {"chronic_burden": 1.0}
    ).set_index("geoid")
    assert out.loc["T0", "raw_need_score"] > out.loc["T1", "raw_need_score"] > out.loc["T2", "raw_need_score"]


def test_missing_acs_factors_renormalize_over_available_weight():
    # Half the weight is on a factor with no ACS data available -- the
    # score should renormalize over just chronic_burden, not silently
    # halve everyone's score.
    tracts = make_tracts((1000, 1000))
    places = make_places([90.0, 10.0], ["T0", "T1"])
    weights = {"chronic_burden": 0.5, "no_vehicle": 0.5}

    with_acs = need_index.compute_need_scores(tracts, places, None, weights).set_index("geoid")
    chronic_only = need_index.compute_need_scores(
        tracts, places, None, {"chronic_burden": 1.0}
    ).set_index("geoid")

    assert with_acs["raw_need_score"].tolist() == pytest.approx(chronic_only["raw_need_score"].tolist())


def test_acs_factors_are_percentile_ranked_not_raw():
    tracts = make_tracts((1000, 1000, 1000))
    places = make_places([0.0, 0.0, 0.0], ["T0", "T1", "T2"])
    acs_factors = pd.DataFrame(
        {"geoid": ["T0", "T1", "T2"], "no_vehicle_pct": [5.0, 50.0, 95.0]}
    )
    out = need_index.compute_need_scores(
        tracts, places, acs_factors, {"no_vehicle": 1.0}
    ).set_index("geoid")
    # Three evenly-spread values should land at roughly 33/67/100th percentile.
    assert out.loc["T2", "raw_need_score"] > out.loc["T1", "raw_need_score"] > out.loc["T0", "raw_need_score"]


def test_normalized_need_is_0_to_100():
    tracts = make_tracts((1000, 2000))
    places = make_places([90.0, 10.0], ["T0", "T1"])
    out = need_index.compute_need_scores(tracts, places, None, {"chronic_burden": 1.0})
    assert out["normalized_need"].max() == pytest.approx(100.0)
    assert out["normalized_need"].min() >= 0.0
