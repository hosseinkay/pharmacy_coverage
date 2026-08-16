"""Census tract boundaries via TIGERweb, for any city given its state +
county FIPS codes. This is the generic counterpart to the local
`chicagotract.geojson` file `ChicagoDataSource` uses -- the piece that
makes a new city a config change (state + county FIPS) instead of needing
a manually-prepared boundary file.

Same WAF quirk as `census_blocks.py`: `f=geojson` gets rejected outright by
this service; `f=json` (Esri JSON) works, hence `arcgis2geojson`.
"""
from __future__ import annotations

import requests
import geopandas as gpd
from arcgis2geojson import arcgis2geojson
from shapely.geometry import shape

TIGERWEB_TRACTS_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "tigerWMS_Current/MapServer/8/query"
)
PAGE_SIZE = 2000
PROJECTED_CRS = "EPSG:3857"


class TigerwebUnavailable(RuntimeError):
    pass


def fetch_tract_geometries(state_fips: str, county_fips: list[str]) -> gpd.GeoDataFrame:
    """(geoid, geometry) in EPSG:3857 for every 2020 Census tract in the
    given counties."""
    county_list = ",".join(f"'{c}'" for c in county_fips)
    where = f"STATE='{state_fips}' AND COUNTY IN ({county_list})"

    geoids: list[str] = []
    geoms = []
    offset = 0

    while True:
        try:
            resp = requests.get(
                TIGERWEB_TRACTS_URL,
                params={
                    "where": where,
                    "outFields": "GEOID",
                    "returnGeometry": "true",
                    "resultRecordCount": PAGE_SIZE,
                    "resultOffset": offset,
                    "f": "json",  # NOT geojson -- see module docstring
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise TigerwebUnavailable(f"TIGERweb request failed: {exc}") from exc

        if "error" in data:
            raise TigerwebUnavailable(f"TIGERweb error: {data['error']}")

        feats = data.get("features", [])
        if not feats:
            break
        for f in feats:
            geoids.append(f["attributes"]["GEOID"])
            geoms.append(shape(arcgis2geojson(f["geometry"])))
        if len(feats) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not geoids:
        raise TigerwebUnavailable("No tracts returned for this state/county selection.")

    return gpd.GeoDataFrame({"geoid": geoids}, geometry=geoms, crs=PROJECTED_CRS)
