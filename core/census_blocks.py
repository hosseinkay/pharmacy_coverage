"""2020 Census block geometry + population.

Why this exists: the coverage math elsewhere (`coverage.residential_per_tract`
+ `CoverageIndex`) assumes a tract's population is spread *uniformly* across
its residential+commercial land -- "newly covered population" is really
"area fraction covered x tract population". That's a reasonable proxy, but
it's not a measured value, and it can be wrong wherever population is
unevenly distributed within a tract.

Census blocks are the finest geography the Census Bureau publishes
population for (typically a few dozen households each). Weighting coverage
by actual block population instead of land area gives a real population
count instead of an area-based guess. See `coverage.BlockCoverageIndex`,
which is a drop-in replacement for `CoverageIndex` once you have blocks.

Two data sources, both free, used together:
  - TIGERweb (geometry): boundaries for 2020 Census blocks, by bounding box.
    Quirk: this service's WAF rejects `f=geojson` requests outright (empty
    "Request Rejected" HTML body) but accepts `f=json` (Esri JSON) fine --
    hence `arcgis2geojson` to convert. Also needs pagination; large bboxes
    return a 500 if requested in one shot.
  - Decennial Census API 2020 PL 94-171 (population): `P1_001N` (total
    population) by block, queryable for an entire county in one request.
"""
from __future__ import annotations

import requests
import pandas as pd
import geopandas as gpd
from arcgis2geojson import arcgis2geojson
from shapely.geometry import shape

from . import acs

TIGERWEB_BLOCKS_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
    "tigerWMS_Current/MapServer/12/query"
)
PAGE_SIZE = 2000
PROJECTED_CRS = "EPSG:3857"


class CensusBlocksUnavailable(RuntimeError):
    pass


def fetch_block_geometries(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """bbox = (minx, miny, maxx, maxy) in EPSG:4326. Returns (geoid,
    geometry) in EPSG:3857, for all 2020 Census blocks intersecting bbox."""
    minx, miny, maxx, maxy = bbox
    geoids: list[str] = []
    geoms = []
    offset = 0

    while True:
        try:
            resp = requests.get(
                TIGERWEB_BLOCKS_URL,
                params={
                    "geometry": f"{minx},{miny},{maxx},{maxy}",
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
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
            raise CensusBlocksUnavailable(f"TIGERweb request failed: {exc}") from exc

        if "error" in data:
            raise CensusBlocksUnavailable(f"TIGERweb error: {data['error']}")

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
        raise CensusBlocksUnavailable("No blocks returned for this bounding box.")

    return gpd.GeoDataFrame({"geoid": geoids}, geometry=geoms, crs=PROJECTED_CRS)


def fetch_block_population(geoids: list[str], api_key: str | None = None) -> pd.DataFrame:
    """Total population (2020 decennial, P1_001N) for every block in the
    counties covering the given tract/block GEOIDs -- one API call per
    county, not per block."""
    api_key = api_key or acs.get_census_api_key()
    if not api_key:
        raise CensusBlocksUnavailable("No CENSUS_API_KEY configured.")

    frames = []
    for state_fips, county_fips in acs.state_county_pairs(geoids):
        try:
            resp = requests.get(
                "https://api.census.gov/data/2020/dec/pl",
                params={
                    "get": "P1_001N",
                    "for": "block:*",
                    "in": f"state:{state_fips}+county:{county_fips}",
                    "key": api_key,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise CensusBlocksUnavailable(f"Census API request failed: {exc}") from exc

        rows = resp.json()
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))

    if not frames:
        raise CensusBlocksUnavailable("No state/county FIPS resolved from GEOIDs.")

    df = pd.concat(frames, ignore_index=True)
    df["geoid"] = df["state"] + df["county"] + df["tract"] + df["block"]
    df["population"] = pd.to_numeric(df["P1_001N"], errors="coerce").fillna(0.0)
    return df[["geoid", "population"]]


def get_population_blocks(
    tract_geoids: list[str], bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Blocks (geometry + population) covering the given tracts, with each
    block's owning tract GEOID attached (the first 11 digits of a 15-digit
    block GEOID are its tract's GEOID -- state+county+tract)."""
    geoms = fetch_block_geometries(bbox)
    pop = fetch_block_population(tract_geoids)

    blocks = geoms.merge(pop, on="geoid", how="left")
    blocks["population"] = blocks["population"].fillna(0.0)
    blocks["tract_geoid"] = blocks["geoid"].str.slice(0, 11)
    blocks = blocks[blocks["tract_geoid"].isin(set(tract_geoids))]
    return blocks[["geoid", "tract_geoid", "population", "geometry"]]
