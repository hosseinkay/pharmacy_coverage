"""Per-city data loading, behind one abstract interface.

This is the seam that makes "different cities" a future add rather than a
rewrite: every other module (need_index, candidates, coverage, optimize,
pipeline) only ever talks to a `CityDataBundle` — never to a city-specific
file format. Adding a new city later means writing one new `CityDataSource`
subclass; nothing else changes.

Only Chicago is wired up and tested in this round (see plan), using the
original report's local data files. Land-use polygons (used to generate
candidate sites) are fetched from OpenStreetMap and disk-cached, since they
generalize to any city by name already.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import cache

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class CityDataBundle:
    tracts: gpd.GeoDataFrame
    """Columns: geoid, TotalPopulation, geometry (tract polygons)."""

    places_scores: pd.DataFrame
    """Columns: geoid, MeasureId, Data_Value (CDC PLACES prevalence rows)."""

    pharmacies: gpd.GeoDataFrame
    """Columns: name, geometry (existing pharmacy points)."""

    zoned_land: gpd.GeoDataFrame
    """Columns: landuse, geometry (residential/commercial polygons)."""

    center: tuple[float, float]
    """(lat, lon) default map center for this city."""


class CityDataSource(ABC):
    name: str

    @abstractmethod
    def load(self) -> CityDataBundle: ...


def _fetch_zoned_land(osm_place_name: str) -> gpd.GeoDataFrame:
    import osmnx as ox

    ox.settings.cache_folder = str(cache.CACHE_DIR / "osmnx_raw")
    ox.settings.use_cache = True

    land = ox.features_from_place(
        osm_place_name, tags={"landuse": ["residential", "commercial"]}
    )
    land = land[land.geometry.type.isin(["Polygon", "MultiPolygon"])]
    land = land[["landuse", "geometry"]].reset_index(drop=True)
    land["landuse"] = land["landuse"].astype(str)
    return land


def load_zoned_land_cached(osm_place_name: str) -> gpd.GeoDataFrame:
    slug = cache.slugify(osm_place_name)
    return cache.cached_geodataframe(
        slug, "zoned_land", lambda: _fetch_zoned_land(osm_place_name)
    )


def _fetch_osm_pharmacies(osm_place_name: str) -> gpd.GeoDataFrame:
    import osmnx as ox

    ox.settings.cache_folder = str(cache.CACHE_DIR / "osmnx_raw")
    ox.settings.use_cache = True

    raw = ox.features_from_place(osm_place_name, tags={"amenity": "pharmacy"})
    raw = raw.reset_index()
    names = raw["name"] if "name" in raw.columns else pd.Series(["Pharmacy"] * len(raw))
    names = names.fillna("Pharmacy")
    # A few pharmacies are tagged on a building footprint (Polygon) rather
    # than a point -- use the centroid so every pharmacy is a Point like
    # the rest of the pipeline expects.
    geoms = raw.geometry.apply(lambda g: g if g.geom_type == "Point" else g.centroid)
    return gpd.GeoDataFrame({"name": names}, geometry=geoms, crs=raw.crs).to_crs("EPSG:4326")


def load_osm_pharmacies_cached(osm_place_name: str) -> gpd.GeoDataFrame:
    slug = cache.slugify(osm_place_name)
    return cache.cached_geodataframe(
        slug, "pharmacies", lambda: _fetch_osm_pharmacies(osm_place_name)
    )


def _extract_lat_lon_from_point_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df["Longitude"] = df[col].str.extract(r"POINT \((.*?) ")[0].astype(float)
    df["Latitude"] = df[col].str.extract(r"POINT \(-?\d+\.\d+ (.*?)\)")[0].astype(float)
    return df


class ChicagoDataSource(CityDataSource):
    name = "Chicago"
    osm_place_name = "Chicago, Illinois, USA"
    center = (41.8781, -87.6298)

    def load(self) -> CityDataBundle:
        tracts = gpd.read_file(DATA_DIR / "chicagotract.geojson")
        tracts["geoid"] = tracts["geoid10"].astype(str)

        places = pd.read_csv(DATA_DIR / "places3.csv")
        places["geoid"] = places["LocationID"].astype(str)

        pop = places[["geoid", "TotalPopulation"]].drop_duplicates("geoid")
        tracts = tracts.merge(pop, on="geoid", how="left")
        tracts["TotalPopulation"] = tracts["TotalPopulation"].fillna(0)
        tracts = tracts[["geoid", "TotalPopulation", "geometry"]]

        pharmacies_df = pd.read_csv(DATA_DIR / "pharmacies.csv")
        pharmacies_df = _extract_lat_lon_from_point_column(
            pharmacies_df, "New Georeferenced Column"
        )
        pharmacies_df = pharmacies_df.dropna(subset=["Latitude", "Longitude"])
        pharmacies = gpd.GeoDataFrame(
            pharmacies_df[["Pharmacy Name"]].rename(columns={"Pharmacy Name": "name"}),
            geometry=gpd.points_from_xy(
                pharmacies_df["Longitude"], pharmacies_df["Latitude"]
            ),
            crs="EPSG:4326",
        )

        zoned_land = load_zoned_land_cached(self.osm_place_name)

        return CityDataBundle(
            tracts=tracts,
            places_scores=places[["geoid", "MeasureId", "Data_Value"]],
            pharmacies=pharmacies,
            zoned_land=zoned_land,
            center=self.center,
        )


class OpenDataCitySource(CityDataSource):
    """Generic CityDataSource for any US city, sourced entirely from public
    APIs instead of a manually-prepared per-city export:
      - tract boundaries: TIGERweb (`tiger_tracts`)
      - health/need data: CDC PLACES public API (`places_api`)
      - existing pharmacies: OSM `amenity=pharmacy` POIs
      - zoned land: OSM `landuse` tags (already generic, see above)

    Adding a city is then just subclassing this with four class attributes
    -- no manual data collection -- which is the entire point of replacing
    Chicago's hand-curated files with open data in the first place.
    """

    name: str = ""
    osm_place_name: str = ""
    state_fips: str = ""
    county_fips: list[str] = []
    center: tuple[float, float] = (0.0, 0.0)

    def load(self) -> CityDataBundle:
        from . import places_api, tiger_tracts

        slug = cache.slugify(self.osm_place_name)
        tract_geoms = cache.cached_geodataframe(
            slug,
            "tracts",
            lambda: tiger_tracts.fetch_tract_geometries(self.state_fips, self.county_fips),
        )
        full_county_fips = [self.state_fips + c for c in self.county_fips]
        places = cache.cached_dataframe(
            slug, "places", lambda: places_api.fetch_places_by_county(full_county_fips)
        )

        pop = places[["geoid", "TotalPopulation"]].drop_duplicates("geoid")
        tracts = tract_geoms.merge(pop, on="geoid", how="left")
        tracts["TotalPopulation"] = tracts["TotalPopulation"].fillna(0)
        tracts = tracts.to_crs("EPSG:4326")[["geoid", "TotalPopulation", "geometry"]]

        pharmacies = load_osm_pharmacies_cached(self.osm_place_name)
        zoned_land = load_zoned_land_cached(self.osm_place_name)

        return CityDataBundle(
            tracts=tracts,
            places_scores=places[["geoid", "MeasureId", "Data_Value"]],
            pharmacies=pharmacies,
            zoned_land=zoned_land,
            center=self.center,
        )


class NewYorkDataSource(OpenDataCitySource):
    name = "New York City"
    osm_place_name = "New York City, New York, USA"
    state_fips = "36"
    county_fips = ["005", "047", "061", "081", "085"]  # Bronx, Kings, New York, Queens, Richmond
    center = (40.7128, -74.0060)


_REGISTRY: dict[str, type[CityDataSource]] = {
    "chicago": ChicagoDataSource,
    "new york city": NewYorkDataSource,
}


def get_data_source(city_key: str) -> CityDataSource:
    try:
        return _REGISTRY[city_key.lower()]()
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown city '{city_key}'. Available: {available}"
        ) from exc


def available_cities() -> list[str]:
    return [cls().name for cls in _REGISTRY.values()]


def get_osm_place_name(city_key: str) -> str:
    return get_data_source(city_key).osm_place_name
