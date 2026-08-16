"""CDC PLACES (Local Data for Better Health, Census Tract Data) via the
public Socrata API. This is the generic, any-city counterpart to the local
`places3.csv` export `ChicagoDataSource` uses -- no Census API key needed,
no manual per-city export, just a county FIPS list.
"""
from __future__ import annotations

import pandas as pd
import requests

PLACES_DATASET_URL = "https://data.cdc.gov/resource/cwsq-ngmh.json"
PAGE_SIZE = 50000


class PlacesUnavailable(RuntimeError):
    pass


def fetch_places_by_county(county_fips: list[str]) -> pd.DataFrame:
    """Returns columns `geoid`, `MeasureId`, `Data_Value`, `TotalPopulation`
    -- the same shape as the local Chicago `places3.csv` export, so
    `need_index.compute_need_scores` doesn't care which city it came from.

    `county_fips` must be the *full* 5-digit state+county FIPS (e.g.
    "36061" for New York County), not just the 3-digit county code.
    """
    fips_list = ",".join(f"'{f}'" for f in county_fips)
    where = f"countyfips in ({fips_list})"

    frames = []
    offset = 0
    while True:
        try:
            resp = requests.get(
                PLACES_DATASET_URL,
                params={
                    "$where": where,
                    "$select": "locationname,measureid,data_value,totalpopulation",
                    "$limit": PAGE_SIZE,
                    "$offset": offset,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise PlacesUnavailable(f"CDC PLACES request failed: {exc}") from exc

        rows = resp.json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not frames:
        raise PlacesUnavailable("No PLACES rows returned for these counties.")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "locationname": "geoid",
            "measureid": "MeasureId",
            "data_value": "Data_Value",
            "totalpopulation": "TotalPopulation",
        }
    )
    df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
    df["TotalPopulation"] = pd.to_numeric(df["TotalPopulation"], errors="coerce")
    return df[["geoid", "MeasureId", "Data_Value", "TotalPopulation"]]
