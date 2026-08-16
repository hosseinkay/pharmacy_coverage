"""Street-network isochrones, for visualization only.

`coverage.py`'s circular buffers are what actually drives the optimizer --
computing a real network isochrone for each of thousands of candidate
sites isn't practical. But the existing-pharmacy coverage picture (a few
hundred to ~1,000 points) and the handful of finally-selected sites (~10)
are cheap enough to do properly, and a real isochrone is a meaningfully
better picture than a circle: it respects rivers, highways, parks, and
dead-end streets instead of pretending travel is as-the-crow-flies.

Uses the *drive* network (road centerlines) rather than the full pedestrian
network as the routing graph. Not because pharmacies are meant to be
reached by car -- the full walk network for an entire city (every
sidewalk and footpath) is large enough to be impractical to hold in memory
on a modest machine. The drive network is far smaller while still
respecting real street topology, which is the part that actually matters
for "is there a river/highway in the way."

Both the network graph and the merged existing-coverage isochrone are
disk-cached per city (the graph fetch + per-point ego_graph computation
for hundreds of points can take a few minutes the first time) -- callers
should expect that latency once per city per radius, not on every rerun.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox

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
