"""Tests for the greedy weighted max-coverage selector.

The key behavior under test is the actual fix over the original model: a
candidate's marginal value must drop once a previously selected candidate
already covered the same need, rather than staying at its static, one-time
reward forever (which is what made the original "MAB" not really react to
its own picks).
"""
import geopandas as gpd
import pytest
from shapely.geometry import Point

from core import optimize

METERS_PER_MILE = 1609.344


def test_greedy_discounts_overlapping_candidates():
    # Candidates 1 and 2 both fully cover the same tract; candidate 3 covers
    # a different, lower-need tract. Far enough apart that the min-distance
    # constraint never kicks in -- only the need-discounting should explain
    # why candidate 2 loses out to candidate 3.
    candidates = gpd.GeoDataFrame(
        {"candidate_id": [1, 2, 3]},
        geometry=[Point(0, 0), Point(10_000, 10_000), Point(20_000, 20_000)],
    )
    candidate_coverage = {1: {"T1": 1.0}, 2: {"T1": 1.0}, 3: {"T2": 1.0}}
    need_scores = {"T1": 100.0, "T2": 50.0}

    result = optimize.greedy_select(
        candidates, candidate_coverage, {}, need_scores,
        num_pharmacies=2, min_distance_miles=0.01,
    )

    assert result.selected_ids == [1, 3]
    assert result.marginal_reward[1] == pytest.approx(100.0)
    assert result.marginal_reward[3] == pytest.approx(50.0)


def test_greedy_respects_min_distance_constraint():
    # Candidate 2 would otherwise be picked (positive marginal gain), but
    # it's within the minimum-distance buffer of candidate 1.
    candidates = gpd.GeoDataFrame(
        {"candidate_id": [1, 2]}, geometry=[Point(0, 0), Point(100, 0)]
    )
    candidate_coverage = {1: {"T1": 0.5}, 2: {"T1": 0.5}}
    need_scores = {"T1": 100.0}

    result = optimize.greedy_select(
        candidates, candidate_coverage, {}, need_scores,
        num_pharmacies=2, min_distance_miles=1.0,  # 1609m, candidates are 100m apart
    )

    assert result.selected_ids == [1]


def test_greedy_matches_brute_force_optimum():
    candidates = gpd.GeoDataFrame(
        {"candidate_id": [1, 2, 3, 4]},
        geometry=[Point(0, 0), Point(50_000, 0), Point(0, 50_000), Point(50_000, 50_000)],
    )
    candidate_coverage = {
        1: {"T1": 0.8, "T2": 0.1},
        2: {"T2": 0.9},
        3: {"T3": 0.7, "T1": 0.2},
        4: {"T1": 0.3, "T3": 0.3},
    }
    need_scores = {"T1": 100.0, "T2": 80.0, "T3": 60.0}

    def total_reward(combo):
        total = 0.0
        for tract, need in need_scores.items():
            uncovered = 1.0
            for cid in combo:
                uncovered *= 1.0 - candidate_coverage[cid].get(tract, 0.0)
            total += need * (1.0 - uncovered)
        return total

    from itertools import combinations

    best_combo = max(combinations([1, 2, 3, 4], 2), key=total_reward)

    result = optimize.greedy_select(
        candidates, candidate_coverage, {}, need_scores,
        num_pharmacies=2, min_distance_miles=0.01,
    )

    assert set(result.selected_ids) == set(best_combo)
