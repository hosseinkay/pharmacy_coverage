"""Census ACS lookups for the tiered desert radius.

The original report's intro cites the literature's actual definition: a
pharmacy desert is a low-income area without a pharmacy within 0.5 miles,
*extended to 1 mile* for low-income communities that have adequate vehicle
access (a car lets you tolerate a farther pharmacy). The original model
never implemented that nuance — it used one fixed 0.5mi buffer everywhere.

This module fetches tract-level median household income and vehicle
availability from the Census ACS 5-Year API and derives a per-tract desert
radius from it. Tract GEOIDs already encode state+county FIPS
(SS+CCC+TTTTTT), so the state/county to query is derived directly from
whatever `CityDataSource` supplies — no per-city hardcoding needed.

Requires a (free) Census API key, read from the `CENSUS_API_KEY` env var
(see `.env.example`). If no key is set or the request fails, callers should
fall back to the fixed radius — this is a UX nicety, not a hard dependency.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ACS_YEAR = 2022
ACS_VARIABLES = ["B19013_001E", "B08201_001E", "B08201_002E"]


class AcsUnavailable(RuntimeError):
    """Raised when ACS data can't be fetched (no key, network error, etc.)."""


def get_census_api_key() -> str | None:
    return os.environ.get("CENSUS_API_KEY") or None


def state_county_pairs(geoids: list[str]) -> set[tuple[str, str]]:
    """Derive (state_fips, county_fips) pairs from 11-digit tract GEOIDs."""
    pairs = set()
    for g in geoids:
        g = str(g).zfill(11)
        pairs.add((g[:2], g[2:5]))
    return pairs


def fetch_tract_acs(geoids: list[str], api_key: str | None = None) -> pd.DataFrame:
    """Fetch median income + vehicle availability for the tracts covering
    the given GEOIDs. Raises AcsUnavailable on any failure."""
    api_key = api_key or get_census_api_key()
    if not api_key:
        raise AcsUnavailable("No CENSUS_API_KEY configured.")

    frames = []
    for state_fips, county_fips in state_county_pairs(geoids):
        url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
        params = {
            "get": ",".join(ACS_VARIABLES),
            "for": "tract:*",
            "in": f"state:{state_fips}+county:{county_fips}",
            "key": api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise AcsUnavailable(f"Census API request failed: {exc}") from exc

        rows = resp.json()
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))

    if not frames:
        raise AcsUnavailable("No state/county FIPS resolved from tract GEOIDs.")

    df = pd.concat(frames, ignore_index=True)
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    for var in ACS_VARIABLES:
        df[var] = pd.to_numeric(df[var], errors="coerce")

    df = df.rename(
        columns={
            "B19013_001E": "median_income",
            "B08201_001E": "total_households",
            "B08201_002E": "households_no_vehicle",
        }
    )
    safe_total = df["total_households"].mask(df["total_households"] == 0)
    df["no_vehicle_share"] = df["households_no_vehicle"] / safe_total
    return df[["geoid", "median_income", "total_households", "households_no_vehicle", "no_vehicle_share"]]


def tiered_desert_radius(
    acs_df: pd.DataFrame,
    base_radius_miles: float = 0.5,
    extended_radius_miles: float = 1.0,
    low_income_pct_of_median: float = 0.8,
    max_no_vehicle_share: float = 0.2,
) -> dict[str, float]:
    """Per-tract desert radius: `extended_radius_miles` for tracts that are
    both low-income (median income below `low_income_pct_of_median` of the
    city-wide tract median) AND have adequate vehicle access (no-vehicle
    household share at or below `max_no_vehicle_share`); `base_radius_miles`
    otherwise. City-relative rather than a fixed dollar threshold, so it
    isn't tied to one region's cost of living.
    """
    valid_income = acs_df["median_income"].where(acs_df["median_income"] > 0)
    city_median = valid_income.median()
    threshold = city_median * low_income_pct_of_median if pd.notna(city_median) else None

    radii: dict[str, float] = {}
    for row in acs_df.itertuples():
        is_low_income = (
            threshold is not None
            and pd.notna(row.median_income)
            and row.median_income > 0
            and row.median_income < threshold
        )
        has_vehicle_access = (
            pd.notna(row.no_vehicle_share) and row.no_vehicle_share <= max_no_vehicle_share
        )
        radii[row.geoid] = (
            extended_radius_miles if (is_low_income and has_vehicle_access) else base_radius_miles
        )
    return radii
