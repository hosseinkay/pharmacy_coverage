"""Decision variables for the pharmacy desert pipeline.

Everything a user should be able to tweak (need-index weights, how many
pharmacies to site, distance constraints, candidate density) lives here as
plain dataclasses, instead of being hardcoded inline across notebook cells
like the original model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Need-index factor weights: score the conditions most tied to *recurring*
# medication access, not every chronic condition CDC PLACES publishes.
# Defaults follow the recommended methodology -- transportation and
# poverty dominate, chronic medication burden (diabetes + high blood
# pressure specifically) close behind, age/mobility next, uninsured rate
# included but deliberately low-weighted (a real barrier, but a weaker
# predictor of *pharmacy* need than the others).
DEFAULT_WEIGHTS: dict[str, float] = {
    "no_vehicle": 0.28,
    "poverty": 0.22,
    "chronic_burden": 0.20,
    "age65": 0.13,
    "mobility": 0.12,
    "uninsured": 0.05,
}

# Human-readable labels + one-line source notes for the sliders in the UI.
FACTOR_LABELS: dict[str, str] = {
    "no_vehicle": "No vehicle access",
    "poverty": "Poverty rate",
    "chronic_burden": "Chronic medication burden (diabetes + high blood pressure)",
    "age65": "Age 65+",
    "mobility": "Mobility / ambulatory disability",
    "uninsured": "Uninsured rate",
}
FACTOR_SOURCES: dict[str, str] = {
    "no_vehicle": "ACS B08201",
    "poverty": "ACS B17001",
    "chronic_burden": "CDC PLACES",
    "age65": "ACS DP05",
    "mobility": "ACS S1810",
    "uninsured": "ACS DP03",
}


# Planning-strategy presets: a non-technical user picks one of these
# instead of starting from 6 raw sliders. "Maximize total residents
# covered" maps to `None` -- the existing pure-population-coverage mode
# (use_health_priorities=False), not a weight dict at all. "Custom
# weights" isn't listed here; it's the UI's escape hatch into the raw
# sliders, seeded from "Balanced need" (DEFAULT_WEIGHTS).
STRATEGY_PRESETS: dict[str, dict[str, float] | None] = {
    "Balanced need (recommended)": dict(DEFAULT_WEIGHTS),
    "Maximize total residents covered": None,
    "Prioritize high-barrier communities": {
        "no_vehicle": 0.35,
        "poverty": 0.30,
        "uninsured": 0.15,
        "chronic_burden": 0.10,
        "age65": 0.05,
        "mobility": 0.05,
    },
    "Prioritize chronic medication burden": {
        "chronic_burden": 0.50,
        "age65": 0.20,
        "mobility": 0.15,
        "no_vehicle": 0.10,
        "poverty": 0.05,
        "uninsured": 0.0,
    },
}
STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "Balanced need (recommended)": "Weighs transportation access, poverty, "
    "chronic medication burden, age, and mobility together -- the "
    "recommended default.",
    "Maximize total residents covered": "Ignores need entirely and treats "
    "every resident the same, choosing sites to cover the most people "
    "regardless of their access barriers.",
    "Prioritize high-barrier communities": "Weighs transportation access, "
    "poverty, and insurance coverage more heavily than chronic disease "
    "rates -- areas where simply *reaching* a pharmacy is the main barrier.",
    "Prioritize chronic medication burden": "Weighs diabetes and high "
    "blood pressure prevalence, plus age 65+ and mobility, more heavily "
    "than transportation or poverty -- areas with the most residents on "
    "recurring prescriptions.",
}
CUSTOM_STRATEGY_LABEL = "Custom weights"


@dataclass
class NeedWeights:
    """Weight assigned to each need-index factor (see DEFAULT_WEIGHTS)."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def normalized(self) -> dict[str, float]:
        """Weights rescaled to sum to 1, so changing one slider doesn't
        silently change the overall magnitude of the need index."""
        total = sum(self.weights.values())
        if total <= 0:
            return {k: 0.0 for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}


@dataclass
class OptConfig:
    """Constraints and parameters for the siting optimization."""

    num_pharmacies: int = 10
    min_distance_miles: float = 0.5
    desert_radius_miles: float = 0.5

    # Tiered desert radius (literature-backed: low-income tracts with
    # adequate vehicle access can tolerate a 1mi radius instead of 0.5mi).
    # Requires a Census API key (`CENSUS_API_KEY` env var); falls back to
    # the fixed `desert_radius_miles` above if unavailable.
    use_tiered_radius: bool = False
    extended_radius_miles: float = 1.0
    low_income_pct_of_median: float = 0.8
    max_no_vehicle_share: float = 0.2

    # Population-weighted coverage (real 2020 Census block population)
    # instead of assuming population is spread evenly across a tract's
    # residential+commercial land. Requires CENSUS_API_KEY; falls back to
    # area-based coverage if unavailable.
    use_population_weighted_coverage: bool = False

    # A policy CONSTRAINT on which sites are even eligible, not a scoring
    # factor: when set, only candidates located inside a federally
    # designated Opportunity Zone are considered at all. Opportunity Zone
    # status never appears in the need index.
    restrict_to_opportunity_zones: bool = False


@dataclass
class CandidateConfig:
    """How candidate pharmacy sites are produced."""

    grid_spacing_meters: float = 200.0
    source: str = "generated"  # "generated" or "csv"
    csv_path: str | None = None
