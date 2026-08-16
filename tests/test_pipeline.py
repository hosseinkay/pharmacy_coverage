import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from core.pipeline import _compute_equity_summary, _reached_per_site

UNIT_SQUARE = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def make_population():
    return {"T1": 1000.0, "T2": 2000.0}


def test_population_reached_discounts_after_first_pick():
    population = make_population()
    candidate_coverage = {
        1: {"T1": 1.0},  # fully covers T1
        2: {"T1": 1.0},  # redundant with site 1
        3: {"T2": 1.0},  # covers the other tract
    }
    reached = _reached_per_site([1, 3], candidate_coverage, {}, population)
    assert reached[1] == pytest.approx(1000.0)
    assert reached[3] == pytest.approx(2000.0)


def test_population_reached_respects_existing_coverage():
    population = make_population()
    candidate_coverage = {1: {"T1": 1.0}}
    # T1 already half-covered by an existing pharmacy.
    reached = _reached_per_site([1], candidate_coverage, {"T1": 0.5}, population)
    assert reached[1] == pytest.approx(500.0)


def test_population_reached_order_matters():
    population = make_population()
    # Both candidates fully cover T1; whichever is picked first gets full
    # credit, the second gets none since nothing's left to reach.
    candidate_coverage = {1: {"T1": 1.0}, 2: {"T1": 1.0}}
    reached = _reached_per_site([1, 2], candidate_coverage, {}, population)
    assert reached[1] == pytest.approx(1000.0)
    assert reached[2] == pytest.approx(0.0)


def test_reached_per_site_works_with_a_zeroed_subset_population():
    # This is exactly how the high-need-only metric is computed: same
    # function, a population dict with non-high-need tracts zeroed out.
    population = {"T1": 1000.0, "T2": 0.0}  # T2 zeroed out (not high-need)
    candidate_coverage = {1: {"T1": 1.0}, 2: {"T2": 1.0}}
    reached = _reached_per_site([1, 2], candidate_coverage, {}, population)
    assert reached[1] == pytest.approx(1000.0)
    assert reached[2] == pytest.approx(0.0)


def make_tracts_with_coverage():
    return gpd.GeoDataFrame(
        {
            "geoid": ["T1", "T2"],
            "TotalPopulation": [1000.0, 1000.0],
            "coverage_before_pct": [0.0, 0.0],
            "coverage_after_pct": [100.0, 0.0],  # only T1 gained coverage
            "geometry": [UNIT_SQUARE, UNIT_SQUARE],
        }
    )


def test_equity_summary_unavailable_without_acs_factors():
    tracts = make_tracts_with_coverage()
    assert _compute_equity_summary(tracts, None) == {"equity_available": False}


def test_equity_summary_attributes_all_new_coverage_to_majority_black_tract():
    tracts = make_tracts_with_coverage()
    acs_factors = pd.DataFrame(
        {"geoid": ["T1", "T2"], "black_share": [0.9, 0.1], "hispanic_share": [0.05, 0.05]}
    )
    equity = _compute_equity_summary(tracts, acs_factors)
    assert equity["equity_available"] is True
    # All newly-covered population is in T1, which is majority-Black.
    assert equity["equity_pct_majority_black"] == pytest.approx(100.0)
    assert equity["equity_pct_majority_hispanic"] == pytest.approx(0.0)


def test_equity_summary_never_reads_race_as_a_weight_input():
    # Race/ethnicity columns being present shouldn't change which tracts
    # the function even looks at for anything other than the reported
    # percentages -- it must not appear anywhere need_index/optimize touch.
    import core.need_index as need_index_module
    import inspect

    source = inspect.getsource(need_index_module)
    assert "black_share" not in source
    assert "hispanic_share" not in source
