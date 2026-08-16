"""ACS-based need factors for the population-weighted need score, plus
race/ethnicity composition for the post-hoc equity validation layer.

The need score deliberately does *not* score every chronic condition CDC
PLACES publishes -- per the recommended methodology, it scores the
factors most tied to *recurring* medication access:
  - vehicle access, poverty, age 65+, and mobility/ambulatory disability
    are demographic/socioeconomic barriers (ACS), not health outcomes
  - "chronic medication burden" stays narrow: diabetes + high blood
    pressure prevalence (CDC PLACES) specifically, since those mean
    ongoing, recurring prescriptions, unlike e.g. a one-time dental visit
    or a smoking habit
  - uninsured rate is included at a deliberately low weight (a financial
    barrier, but a weaker predictor of *pharmacy* need specifically than
    the others)

Race/ethnicity is intentionally *not* a scoring input. It's fetched here
so callers can report it as an equity check after the fact -- "what share
of newly covered residents are in majority-Black or majority-Latino
tracts" -- without making race itself a variable the optimizer optimizes
on. See pipeline.py's equity summary fields.
"""
from __future__ import annotations

import pandas as pd
import requests

from . import acs

ACS_YEAR = 2022
SENTINEL_THRESHOLD = -1_000_000  # Census uses several negative codes (-666666666 etc.) for "not computable"


class NeedFactorsUnavailable(RuntimeError):
    pass


def _get(url: str, params: dict) -> pd.DataFrame:
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise NeedFactorsUnavailable(f"ACS request failed: {exc}") from exc
    rows = resp.json()
    return pd.DataFrame(rows[1:], columns=rows[0])


def _clean(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mask(numeric <= SENTINEL_THRESHOLD)


def fetch_need_factors(geoids: list[str], api_key: str | None = None) -> pd.DataFrame:
    """Per-tract ACS factors, as percentages (0-100) ready for percentile
    ranking, plus race/ethnicity shares (0-1) for the equity layer.

    Columns: geoid, no_vehicle_pct, poverty_pct, age65_pct,
    ambulatory_disability_pct, uninsured_pct, black_share, hispanic_share.
    """
    api_key = api_key or acs.get_census_api_key()
    if not api_key:
        raise NeedFactorsUnavailable("No CENSUS_API_KEY configured.")

    frames = []
    for state_fips, county_fips in acs.state_county_pairs(geoids):
        in_clause = f"state:{state_fips}+county:{county_fips}"

        detail = _get(
            f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5",
            {
                "get": "B08201_001E,B08201_002E,B17001_001E,B17001_002E,"
                "B03002_001E,B03002_004E,B03002_012E",
                "for": "tract:*",
                "in": in_clause,
                "key": api_key,
            },
        )
        profile = _get(
            f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5/profile",
            {"get": "DP05_0024PE,DP03_0099PE", "for": "tract:*", "in": in_clause, "key": api_key},
        )
        subject = _get(
            f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5/subject",
            {"get": "S1810_C03_047E", "for": "tract:*", "in": in_clause, "key": api_key},
        )

        merged = detail.merge(profile, on=["state", "county", "tract"]).merge(
            subject, on=["state", "county", "tract"]
        )
        frames.append(merged)

    if not frames:
        raise NeedFactorsUnavailable("No ACS rows returned for these tracts.")

    df = pd.concat(frames, ignore_index=True)
    df["geoid"] = df["state"] + df["county"] + df["tract"]

    for col in [
        "B08201_001E", "B08201_002E", "B17001_001E", "B17001_002E",
        "B03002_001E", "B03002_004E", "B03002_012E",
        "DP05_0024PE", "DP03_0099PE", "S1810_C03_047E",
    ]:
        df[col] = _clean(df[col])

    safe_households = df["B08201_001E"].mask(df["B08201_001E"] == 0)
    safe_poverty_universe = df["B17001_001E"].mask(df["B17001_001E"] == 0)
    safe_total_pop = df["B03002_001E"].mask(df["B03002_001E"] == 0)

    return pd.DataFrame(
        {
            "geoid": df["geoid"],
            "no_vehicle_pct": (df["B08201_002E"] / safe_households) * 100.0,
            "poverty_pct": (df["B17001_002E"] / safe_poverty_universe) * 100.0,
            "age65_pct": df["DP05_0024PE"],
            "ambulatory_disability_pct": df["S1810_C03_047E"],
            "uninsured_pct": df["DP03_0099PE"],
            "black_share": df["B03002_004E"] / safe_total_pop,
            "hispanic_share": df["B03002_012E"] / safe_total_pop,
        }
    )
