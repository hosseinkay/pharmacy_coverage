"""Pharmacy Need Index: a population-weighted composite of the factors
most tied to *recurring* medication access, not a sum of every chronic
condition CDC PLACES happens to publish.

Score = Population x weighted_sum(percentile_rank(factor) for each factor)

Factors (see core/acs_need_factors.py for sourcing):
  - no_vehicle:      % of households without a vehicle (ACS) -- transportation barrier
  - poverty:         poverty rate (ACS) -- economic vulnerability
  - chronic_burden:  mean of diabetes + high blood pressure prevalence
                      (CDC PLACES) -- the two conditions that mean ongoing,
                      recurring prescriptions, not every health outcome
  - age65:           % age 65+ (ACS) -- older-adult medication demand
  - mobility:         % with an ambulatory difficulty (ACS) -- physical
                      access barrier to getting to a pharmacy at all
  - uninsured:        uninsured rate (ACS) -- financial barrier, weighted
                      low since it's a weaker predictor of pharmacy need
                      specifically than the others

Each factor is percentile-ranked (0-100) *within this city's tracts*
before combining, so a vehicle-ownership percentage, a prevalence rate,
and a poverty rate -- all different units and scales -- are comparable.
Obesity, smoking, depression, asthma, COPD, high cholesterol, dental
visits, and vision/general disability are deliberately excluded: they
don't reliably predict ongoing pharmacy need the way diabetes/hypertension
medication adherence does, and including them just adds noise.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

CHRONIC_BURDEN_MEASURES = ["DIABETES", "BPHIGH"]

# Maps each weight key to the column in acs_need_factors.fetch_need_factors's
# output (chronic_burden is computed separately from CDC PLACES, not ACS).
ACS_FACTOR_COLUMNS = {
    "no_vehicle": "no_vehicle_pct",
    "poverty": "poverty_pct",
    "age65": "age65_pct",
    "mobility": "ambulatory_disability_pct",
    "uninsured": "uninsured_pct",
}


def percentile_rank(series: pd.Series) -> pd.Series:
    """0-100 rank of each value relative to the others in `series`."""
    if series.dropna().empty:
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True, na_option="bottom") * 100.0


def compute_need_scores(
    tracts: gpd.GeoDataFrame,
    places_scores: pd.DataFrame,
    acs_factors: pd.DataFrame | None,
    weights: dict[str, float],
    use_health_priorities: bool = True,
) -> pd.DataFrame:
    """Compute per-tract pharmacy need.

    Args:
        tracts: must contain columns `geoid`, `TotalPopulation`.
        places_scores: must contain columns `geoid`, `MeasureId`, `Data_Value`
            (used only for the chronic_burden factor).
        acs_factors: output of `acs_need_factors.fetch_need_factors`, or
            None if ACS is unavailable -- factors needing it just drop out
            of the composite (with the remaining weights renormalized)
            rather than failing the whole score.
        weights: factor name -> weight (no_vehicle, poverty, chronic_burden,
            age65, mobility, uninsured). Should already be normalized to
            sum to 1 by `NeedWeights.normalized()`, but this function
            renormalizes over whatever factors are actually available.
        use_health_priorities: if False, every resident counts equally
            (raw_need_score = 1 for every tract), so siting is driven
            purely by population coverage.

    Returns:
        DataFrame indexed by `geoid` with columns:
            raw_need_score  -- the 0-100ish percentile composite (or flat
                               1.0 if use_health_priorities=False)
            population
            weighted_need   -- raw_need_score * population
            normalized_need -- weighted_need scaled 0-100, for display/coloring only
    """
    pop = (
        tracts[["geoid", "TotalPopulation"]]
        .drop_duplicates("geoid")
        .set_index("geoid")["TotalPopulation"]
        .rename("population")
    )

    if use_health_priorities and weights:
        factor_table = pd.DataFrame(index=pop.index)

        chronic = (
            places_scores[places_scores["MeasureId"].isin(CHRONIC_BURDEN_MEASURES)]
            .groupby("geoid")["Data_Value"]
            .mean()
        )
        factor_table["chronic_burden"] = chronic

        if acs_factors is not None:
            acs_indexed = acs_factors.set_index("geoid")
            for factor, column in ACS_FACTOR_COLUMNS.items():
                if column in acs_indexed.columns:
                    factor_table[factor] = acs_indexed[column]

        raw = pd.Series(0.0, index=pop.index)
        weight_used = 0.0
        for factor, weight in weights.items():
            if weight <= 0 or factor not in factor_table.columns:
                continue
            raw = raw.add(percentile_rank(factor_table[factor]) * weight, fill_value=0.0)
            weight_used += weight
        if weight_used > 0:
            raw = raw / weight_used  # renormalize over only the factors actually available

        out = pop.to_frame()
        out["raw_need_score"] = raw
    else:
        out = pop.to_frame()
        out["raw_need_score"] = 1.0

    out["population"] = out["population"].fillna(0.0)
    out["raw_need_score"] = out["raw_need_score"].fillna(0.0)
    out["weighted_need"] = out["raw_need_score"] * out["population"]

    max_val = out["weighted_need"].max()
    out["normalized_need"] = (
        0.0 if not max_val or max_val <= 0 else (out["weighted_need"] / max_val) * 100.0
    )

    return out.reset_index()
