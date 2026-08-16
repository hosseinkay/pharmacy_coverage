import geopandas as gpd
from shapely.geometry import Point, Polygon

from core import candidates as cand_mod

ZONE = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])


def make_zoned_land():
    return gpd.GeoDataFrame({"landuse": ["residential"]}, geometry=[ZONE], crs="EPSG:3857")


def test_generate_grid_candidates_is_deterministic():
    land = make_zoned_land()
    a = cand_mod.generate_grid_candidates(land, spacing_meters=100.0)
    b = cand_mod.generate_grid_candidates(land, spacing_meters=100.0)

    assert len(a) > 0
    coords_a = sorted((round(g.x, 3), round(g.y, 3)) for g in a.geometry)
    coords_b = sorted((round(g.x, 3), round(g.y, 3)) for g in b.geometry)
    assert coords_a == coords_b


def test_generate_grid_candidates_stays_within_zoned_land():
    land = make_zoned_land()
    cands = cand_mod.generate_grid_candidates(land, spacing_meters=100.0)
    for geom in cands.geometry:
        assert ZONE.intersects(geom)


def test_exclude_near_existing_respects_min_distance():
    land = make_zoned_land()
    cands = cand_mod.generate_grid_candidates(land, spacing_meters=100.0)

    pharmacy_point = Point(500, 500)
    pharmacies = gpd.GeoDataFrame({"name": ["P1"]}, geometry=[pharmacy_point], crs="EPSG:3857")

    min_distance_miles = 0.05
    filtered = cand_mod.exclude_near_existing(cands, pharmacies, min_distance_miles)

    assert 0 < len(filtered) < len(cands)
    min_dist_m = min_distance_miles * cand_mod.METERS_PER_MILE
    for geom in filtered.geometry:
        assert geom.distance(pharmacy_point) >= min_dist_m - 1e-6
