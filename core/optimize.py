"""Greedy weighted maximum-coverage site selection.

This replaces the original "Multi-Armed Bandit": that model computed each
candidate's reward once, before any selection happened, so its
epsilon-greedy loop just repeatedly sampled a fixed value and converged to
sorting by reward — a candidate's score never dropped after a *previous*
candidate already covered the same tracts (only a separate min-distance
filter prevented clustering).

What's actually being solved here is the classic weighted maximum coverage
location problem. The standard, well-understood approach is a greedy
algorithm: pick the candidate with the highest marginal gain in *still
uncovered* weighted need, discount that need by how much the pick just
covered, and repeat. This has a known 1-1/e (~63%) optimality guarantee, is
deterministic, and is fast enough to rerun on every UI change.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd

METERS_PER_MILE = 1609.344


@dataclass
class SelectionResult:
    selected_ids: list
    marginal_reward: dict  # candidate_id -> weighted gain at the time it was picked


def greedy_select(
    candidates: gpd.GeoDataFrame,
    candidate_coverage: dict[int, dict[str, float]],
    existing_coverage: dict[str, float],
    need_scores: dict[str, float],
    num_pharmacies: int,
    min_distance_miles: float,
) -> SelectionResult:
    """
    Args:
        candidates: GeoDataFrame with `candidate_id` and projected `geometry`.
        candidate_coverage: candidate_id -> {geoid: fraction of that tract's
            residential area covered if this candidate is sited}.
        existing_coverage: geoid -> fraction already covered by existing pharmacies.
        need_scores: geoid -> total weighted need (population-scaled).
        num_pharmacies: how many sites to select (B).
        min_distance_miles: minimum spacing between selected sites.

    Returns:
        SelectionResult with the chosen candidate ids in selection order and
        the marginal reward each contributed at the time it was picked.
    """
    remaining_need = {
        geoid: need_scores.get(geoid, 0.0) * (1.0 - existing_coverage.get(geoid, 0.0))
        for geoid in need_scores
    }

    min_dist_m = min_distance_miles * METERS_PER_MILE
    geoms = dict(zip(candidates["candidate_id"], candidates.geometry))

    pool = set(candidate_coverage.keys())
    selected: list = []
    rewards: dict = {}

    for _ in range(num_pharmacies):
        best_id, best_gain = None, 0.0
        for cid in pool:
            if any(geoms[cid].distance(geoms[s]) < min_dist_m for s in selected):
                continue
            gain = sum(
                remaining_need.get(geoid, 0.0) * frac
                for geoid, frac in candidate_coverage[cid].items()
            )
            if gain > best_gain:
                best_id, best_gain = cid, gain

        if best_id is None or best_gain <= 0:
            break

        selected.append(best_id)
        rewards[best_id] = best_gain
        pool.discard(best_id)
        for geoid, frac in candidate_coverage[best_id].items():
            if geoid in remaining_need:
                remaining_need[geoid] *= (1.0 - frac)

    return SelectionResult(selected_ids=selected, marginal_reward=rewards)
