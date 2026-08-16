"""BlockCoverageIndex should give a meaningfully different (and more
correct) answer than the area-based CoverageIndex when population is
unevenly distributed within a tract -- that's the whole reason it exists.
"""
import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from core.coverage import METERS_PER_MILE, BlockCoverageIndex, CoverageIndex, merged_buffer, nearest_distance_miles

DENSE_BLOCK = Polygon([(1000, 0), (2000, 0), (2000, 1000), (1000, 1000)])
SPARSE_BLOCK = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
RADIUS_MILES = 600 / 1609.344  # 600m, centered in the sparse block


def test_population_weighting_differs_from_area_weighting():
    blocks = gpd.GeoDataFrame(
        {"geoid": ["B1", "B2"], "tract_geoid": ["T1", "T1"], "population": [900.0, 100.0]},
        geometry=[DENSE_BLOCK, SPARSE_BLOCK],
        crs="EPSG:3857",
    )
    pop_weighted = BlockCoverageIndex(blocks).coverage_fractions(Point(500, 500), RADIUS_MILES)

    # The area-based index doesn't know about population at all -- treat
    # the same two-block footprint as one tract for a fair comparison.
    tract_residential = gpd.GeoDataFrame(
        {"geoid": ["T1"], "residential_area": [DENSE_BLOCK.union(SPARSE_BLOCK).area]},
        geometry=[DENSE_BLOCK.union(SPARSE_BLOCK)],
        crs="EPSG:3857",
    )
    area_weighted = CoverageIndex(tract_residential).coverage_fractions(Point(500, 500), RADIUS_MILES)

    # The buffer mostly sits in the sparse (100-person) block and barely
    # touches the dense (900-person) block, so population-weighted coverage
    # should be substantially lower than the area-based number, which has
    # no way to know most of the covered land is sparsely populated.
    assert pop_weighted["T1"] < area_weighted["T1"] * 0.5


def test_coverage_is_population_proportion_not_area_proportion():
    # A buffer covering 100% of a tiny, sparsely-populated block and 0% of
    # everything else should report coverage equal to that block's share of
    # total tract population -- not its share of tract area.
    tiny_block = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # area = 100
    huge_block = Polygon([(1000, 1000), (2000, 1000), (2000, 2000), (1000, 2000)])  # area = 1,000,000
    blocks = gpd.GeoDataFrame(
        {"geoid": ["small", "big"], "tract_geoid": ["T1", "T1"], "population": [10.0, 990.0]},
        geometry=[tiny_block, huge_block],
        crs="EPSG:3857",
    )
    index = BlockCoverageIndex(blocks)
    # 80m radius: comfortably covers all of tiny_block (farthest corner is
    # ~7m from center) but falls ~1.3km short of huge_block.
    frac = index.coverage_fractions(Point(5, 5), radius_miles=80 / 1609.344)
    assert frac["T1"] == pytest.approx(10.0 / 1000.0, abs=0.01)


def test_merged_buffer_empty_points_returns_none():
    empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:3857")
    assert merged_buffer(empty, 0.5) is None


def test_merged_buffer_unions_overlapping_circles():
    # Two points 100m apart with a 1-mile radius overlap heavily -- the
    # union's area must be less than the sum of two full circles (otherwise
    # we're double-counting the overlap), but more than either circle alone.
    points = gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0, 0), Point(100, 0)], crs="EPSG:3857"
    )
    radius_m = 1.0 * METERS_PER_MILE
    single_circle_area = Point(0, 0).buffer(radius_m).area

    merged = merged_buffer(points, 1.0)

    assert merged.area > single_circle_area
    assert merged.area < 2 * single_circle_area


def test_merged_buffer_disjoint_points_sum_areas():
    # Far enough apart that the buffers don't touch -- union area should
    # equal the sum of the two individual circles (no overlap to dedupe).
    points = gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[Point(0, 0), Point(1_000_000, 0)], crs="EPSG:3857"
    )
    radius_m = 0.5 * METERS_PER_MILE
    single_circle_area = Point(0, 0).buffer(radius_m).area

    merged = merged_buffer(points, 0.5)

    assert merged.area == pytest.approx(2 * single_circle_area, rel=0.01)


def test_nearest_distance_picks_the_closer_target():
    points = gpd.GeoDataFrame({"id": ["a", "b"]}, geometry=[Point(0, 0), Point(1000, 1000)], crs="EPSG:3857")
    targets = gpd.GeoDataFrame({"name": ["near", "far"]}, geometry=[Point(10, 0), Point(2000, 2000)], crs="EPSG:3857")

    distances = nearest_distance_miles(points, targets)

    assert distances.loc[0] == pytest.approx(10 / METERS_PER_MILE)
    assert distances.loc[1] == pytest.approx(1000 * (2 ** 0.5) / METERS_PER_MILE, rel=0.01)


def test_nearest_distance_empty_targets_returns_nan():
    points = gpd.GeoDataFrame({"id": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:3857")
    empty_targets = gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:3857")

    distances = nearest_distance_miles(points, empty_targets)

    assert distances.isna().all()
