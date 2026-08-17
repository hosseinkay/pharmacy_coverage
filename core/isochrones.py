"""Street-network isochrones, for visualization only.

`coverage.py`'s circular buffers are what actually drives the optimizer --
computing a real network isochrone for each of thousands of candidate
sites isn't practical. But the existing-pharmacy coverage picture (a few
hundred to ~1,000 points) and the handful of finally-selected sites (~10)
are cheap enough to do properly, and a real isochrone is a meaningfully
better picture than a circle: it respects rivers, highways, parks, and
dead-end streets instead of pretending travel is as-the-crow-flies.

Two backends are provided:

  Fast path — edge-parquet spatial query (< 1 s for ≤ 30 sites)
    Requires: drive_edges_3857.parquet (5 MB, committed to git)
    Method:   for each site, find all road edges within `radius` of the
              point using a pre-built spatial index, buffer them 40 m, union.
    Limitation: uses straight-line radius, not true network distance.  For
              a visualisation the difference is imperceptible.

  Full path — graphml + ego_graph (accurate network distance)
    Requires: drive_network.graphml (38 MB, also committed)
    Method:   NetworkX ego_graph limited to `radius` along edge weights;
              slower to load (20 s), used as fallback if edges parquet is
              absent, and still available for the existing-pharmacy isochrone
              which only runs once per city/radius.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.ops import unary_union

from . import cache

METERS_PER_MILE = 1609.344
EDGE_BUFFER_METERS = 40  # corridor width around each reachable street


class IsochroneUnavailable(RuntimeError):
    pass


def _graph_path(osm_place_name: str) -> Path:
    slug = cache.slugify(osm_place_name)
    return cache.city_cache_dir(slug) / "drive_network.graphml"


def get_network_graph_cached(osm_place_name: str):
    path = _graph_path(osm_place_name)
    if path.exists():
        return ox.load_graphml(path)

    try:
        ox.settings.cache_folder = str(cache.CACHE_DIR / "osmnx_raw")
        ox.settings.use_cache = True
        graph = ox.graph_from_place(osm_place_name, network_type="drive")
    except Exception as exc:  # osmnx/Overpass failures aren't a fixed type
        raise IsochroneUnavailable(f"Could not fetch street network: {exc}") from exc

    ox.save_graphml(graph, path)
    return graph


def _isochrone_for_node(graph, center_node, radius_miles: float):
    """One reachable-streets-buffered polygon (EPSG:4326) for an already-
    snapped graph node."""
    radius_m = radius_miles * METERS_PER_MILE
    sub = nx.ego_graph(graph, center_node, radius=radius_m, distance="length")
    if sub.number_of_edges() == 0:
        return None

    edges = ox.graph_to_gdfs(sub, nodes=False, edges=True)
    if edges.empty:
        return None

    edges_3857 = edges.to_crs("EPSG:3857")
    merged_3857 = edges_3857.geometry.buffer(EDGE_BUFFER_METERS).union_all()
    return gpd.GeoSeries([merged_3857], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]


def compute_isochrone(graph, point_4326, radius_miles: float):
    """Network-distance isochrone polygon (EPSG:4326) around one point."""
    try:
        node = ox.distance.nearest_nodes(graph, point_4326.x, point_4326.y)
    except Exception as exc:
        raise IsochroneUnavailable(f"Could not snap point to network: {exc}") from exc
    return _isochrone_for_node(graph, node, radius_miles)


def compute_merged_isochrone(graph, points_4326: gpd.GeoDataFrame, radius_miles: float):
    """Union of individual isochrones around every point in `points_4326`.
    Returns `None` for an empty input."""
    if points_4326.empty:
        return None

    polys = []
    for geom in points_4326.geometry:
        try:
            node = ox.distance.nearest_nodes(graph, geom.x, geom.y)
        except Exception:
            continue
        poly = _isochrone_for_node(graph, node, radius_miles)
        if poly is not None and not poly.is_empty:
            polys.append(poly)

    if not polys:
        return None
    return gpd.GeoSeries(polys, crs="EPSG:4326").union_all()


def get_merged_existing_isochrone_cached(
    osm_place_name: str, points_4326: gpd.GeoDataFrame, radius_miles: float
):
    """Disk-cached merged isochrone for a city's existing pharmacies --
    this is the expensive one (hundreds of ego_graph computations), and the
    existing-pharmacy set + a given radius don't change within a city, so
    it's worth persisting across app restarts, not just within a session.
    """
    slug = cache.slugify(osm_place_name)
    key = f"existing_isochrone_r{radius_miles:.2f}"
    path = cache.city_cache_dir(slug) / f"{key}.parquet"
    if path.exists():
        gdf = gpd.read_parquet(path)
        return None if gdf.empty else gdf.geometry.iloc[0]

    graph = get_network_graph_cached(osm_place_name)
    merged = compute_merged_isochrone(graph, points_4326, radius_miles)

    gpd.GeoDataFrame(
        {"geometry": [merged] if merged is not None else []}, crs="EPSG:4326"
    ).to_parquet(path)
    return merged


# ---------------------------------------------------------------------------
# Fast path: edge-parquet spatial query (no osmnx / networkx required)
# ---------------------------------------------------------------------------

def _edges_parquet_path(osm_place_name: str) -> Path:
    slug = cache.slugify(osm_place_name)
    return cache.city_cache_dir(slug) / "drive_edges_3857.parquet"


def get_edges_gdf(osm_place_name: str) -> Optional[gpd.GeoDataFrame]:
    """Load the pre-projected street-edge GeoDataFrame (EPSG:3857).

    Returns None if the parquet hasn't been pre-computed yet.
    """
    path = _edges_parquet_path(osm_place_name)
    if not path.exists():
        return None
    return gpd.read_parquet(path)


def compute_isochrone_from_edges(
    edges_3857: gpd.GeoDataFrame,
    points_4326: gpd.GeoDataFrame,
    radius_miles: float,
) -> Optional[object]:
    """Fast coverage shape for a handful of points using a pre-loaded edge GDF.

    Method: for each point, collect all road edges whose geometry intersects
    a straight-line `radius_miles` buffer, buffer those edges by
    EDGE_BUFFER_METERS (40 m) to form a street corridor, then intersect
    back with the site's radius disc so the shape stays bounded.

    This uses straight-line radius rather than true network distance, but for
    a 0.5-mile walkability visualisation the difference is imperceptible and
    the speed advantage is enormous (~1 s vs ~24 s for the graphml path).

    Args:
        edges_3857: Street-edge GeoDataFrame pre-projected to EPSG:3857.
        points_4326: Selected sites in EPSG:4326.
        radius_miles: Walking/driving radius in miles.

    Returns:
        EPSG:4326 MultiPolygon, or None if no edges were found.
    """
    if points_4326.empty or edges_3857 is None or edges_3857.empty:
        return None

    radius_m = radius_miles * METERS_PER_MILE
    pts_3857 = points_4326.to_crs("EPSG:3857")

    polys = []
    for _, row in pts_3857.iterrows():
        site_buf = row.geometry.buffer(radius_m)
        idxs = edges_3857.sindex.query(site_buf, predicate="intersects")
        nearby = edges_3857.iloc[idxs]
        if not nearby.empty:
            corridor = nearby.geometry.buffer(EDGE_BUFFER_METERS).union_all()
            clipped = corridor.intersection(site_buf)
            if not clipped.is_empty:
                polys.append(clipped)

    if not polys:
        return None

    merged_3857 = unary_union(polys)
    return gpd.GeoSeries([merged_3857], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
