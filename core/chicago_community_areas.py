"""Chicago's official 77 Community Areas.

Census tracts are precise but not how Chicagoans actually refer to
neighborhoods -- "Austin" or "Hyde Park" means more to a reader than a
tract GEOID. This is intentionally Chicago-specific (the city publishes
these as an open dataset; other cities don't have the exact same concept),
so it's used as an optional enrichment layer, not a replacement for the
tract-based math elsewhere in the app.
"""
from __future__ import annotations

import geopandas as gpd
import requests
from shapely.geometry import shape

COMMUNITY_AREAS_URL = "https://data.cityofchicago.org/resource/igwz-8jzy.json"


class CommunityAreasUnavailable(RuntimeError):
    pass


def fetch_community_areas() -> gpd.GeoDataFrame:
    """Columns: community_area (title-cased name), geometry (EPSG:4326)."""
    try:
        resp = requests.get(COMMUNITY_AREAS_URL, params={"$limit": 200}, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except requests.RequestException as exc:
        raise CommunityAreasUnavailable(f"Community areas request failed: {exc}") from exc

    if not rows:
        raise CommunityAreasUnavailable("No community areas returned.")

    names = [row["community"].title() for row in rows]
    geoms = [shape(row["the_geom"]) for row in rows]
    return gpd.GeoDataFrame({"community_area": names}, geometry=geoms, crs="EPSG:4326")
