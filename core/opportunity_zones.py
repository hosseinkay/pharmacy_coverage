"""Federally-designated Qualified Opportunity Zones (2017 Tax Cuts and Jobs
Act), via HUD's public ArcGIS feature service.

Unlike Community Areas (Chicago-only), Opportunity Zone designation is
itself a census-tract attribute, so this works for any city -- just look up
which of a city's tract GEOIDs are on the designated list.
"""
from __future__ import annotations

import requests

from . import acs

OZ_QUERY_URL = (
    "https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/"
    "Opportunity_Zones/FeatureServer/13/query"
)
PAGE_SIZE = 2000


class OpportunityZonesUnavailable(RuntimeError):
    pass


def fetch_designated_tracts(geoids: list[str]) -> set[str]:
    """Returns the subset of 11-digit tract GEOIDs in `geoids` that are
    designated Opportunity Zones."""
    designated: set[str] = set()
    for state_fips, county_fips in acs.state_county_pairs(geoids):
        offset = 0
        while True:
            try:
                resp = requests.get(
                    OZ_QUERY_URL,
                    params={
                        "where": f"STATE='{state_fips}' AND COUNTY='{county_fips}'",
                        "outFields": "GEOID10",
                        "returnGeometry": "false",
                        "resultRecordCount": PAGE_SIZE,
                        "resultOffset": offset,
                        "f": "json",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                raise OpportunityZonesUnavailable(f"OZ request failed: {exc}") from exc

            if "error" in data:
                raise OpportunityZonesUnavailable(f"OZ error: {data['error']}")

            feats = data.get("features", [])
            if not feats:
                break
            designated.update(f["attributes"]["GEOID10"] for f in feats)
            if len(feats) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    return designated
