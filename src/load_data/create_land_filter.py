"""
Creates the French landmass shapefile.
"""

import io
import os
import geopandas as gpd
from shapely.ops import linemerge
from shapely.ops import polygonize
from shapely.ops import unary_union
import requests
import zipfile
import utm
import pyproj

import src.file_utils as file_utils


def load_data() -> None:
    """
    Saves the European coastline and French territory shapefiles locally.
    """
    root_dir = file_utils.get_root_path()
    output_dir = f'{root_dir}/data/shapefiles'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # European coastline (https://www.eea.europa.eu/ds_resolveuid/06227e40310045408ac8be0d469e1189)
    link = 'https://www.eea.europa.eu/data-and-maps/data/eea-coastline-for-analysis-1/gis-data/europe-coastline-shapefile/at_download/file'
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.join(output_dir, 'raw', 'Europe_coastline_shapefile'))

    # French sea
    # This shapefile needs to be downloaded manually from this location:
    # https://www.marineregions.org/gazetteer.php?p=details&id=5677
    # Download as a shapefile and save to output_dir such that the file path is data/shapefiles/raw/eez/eez.shp

    # French territory
    link = 'https://www.eea.europa.eu/data-and-maps/data/eea-reference-grids-2/gis-files/france-shapefile/at_download/file'
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.join(output_dir, 'raw', 'France_shapefile'))


def get_utm_zone_from_lat_lon(
    lat: float,
    lon: float,
) -> pyproj.CRS:
    """Generate UTM CRS from latitude longitude pair.
    From reglab-utils (spatial)

    Args:
        lat: latitude
        lon: longitude

    Returns:
        PyProj CRS object
    """

    projection_dict = {
        "proj": "utm",
        "zone": utm.latlon_to_zone_number(lat, lon),
    }
    if lat < 0:
        projection_dict["south"] = True
    return pyproj.CRS.from_dict(projection_dict)


def main():
    root_dir = file_utils.get_root_path()
    os.makedirs(f"{root_dir}/data/shapefiles/clean", exist_ok=True)
    load_data()

    # Read in datasets
    eu_coastline = gpd.read_file(f"{root_dir}/data/shapefiles/raw/Europe_coastline_shapefile/Europe_coastline.shp").to_crs(
        "EPSG:4326")
    eu_coastline_part = gpd.GeoDataFrame(
        geometry=[line for i, line in enumerate(eu_coastline.geometry[0].geoms) if i < 10000])
    french_sea = gpd.read_file(f"{root_dir}/data/shapefiles/raw/eez/eez.shp").to_crs("EPSG:4326")
    france_shape = gpd.GeoDataFrame(
        geometry=[gpd.read_file(f"{root_dir}/data/shapefiles/raw/France_shapefile/fr_10km.shp").to_crs("EPSG:4326").unary_union],
        crs="EPSG:4326"
    )

    utm_crs = get_utm_zone_from_lat_lon(france_shape.geometry[0].centroid.y, france_shape.geometry[0].centroid.x)

    # Obtain Mediterranean Sea
    # * Remove overlap of French sea from French shape
    # * Dissect remaining France shape using French coastline shape
    # (obtained from intersection of France shape and EU coastline shape).
    # * Keep parts of shape representing the Mediterranean sea.
    france_no_marine = france_shape.overlay(french_sea, how="difference")
    france_coastline = eu_coastline.overlay(france_shape, how="intersection")

    # Get polygons of all EU geoms so we can later recover islands we lose when picking largest results from dissection.
    line_split_collection = [linestring for linestring in eu_coastline.geometry[0].geoms]
    lines = unary_union(line_split_collection)
    lines = linemerge(lines)
    polygons = list(polygonize(lines))
    eu_geoms = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # Get list of France shape and coastline boundaries. Then dissect France using France coastline
    line_split_collection = [linestring for linestring in france_coastline.geometry[0].geoms]
    line_split_collection.append(france_no_marine.geometry[0].boundary)
    lines = unary_union(line_split_collection)
    lines = linemerge(lines)
    polygons = list(polygonize(lines))
    dissected = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # Sort dissected France polygons by area and find areas corresponding to Mediterranean sea.
    dissected["area"] = dissected.to_crs(utm_crs).area
    dissected["perimeter"] = dissected.to_crs(utm_crs).length

    # Obtain French primary landmass
    no_marine_intermediate = dissected.sort_values('area', ascending=False).iloc[0:2]

    line_split_collection = [linestring for linestring in france_coastline.geometry[0].geoms]
    line_split_collection.append(no_marine_intermediate.geometry.iloc[0].boundary)
    lines = unary_union(line_split_collection)
    lines = linemerge(lines)
    polygons = list(polygonize(lines))
    intermediate_poly = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    intermediate_poly['area'] = intermediate_poly.to_crs(utm_crs).area
    intermediate_poly['permiter'] = intermediate_poly.to_crs(utm_crs).length
    intermediate_poly.sort_values('area', ascending=False).iloc[0:6]

    # From plotting we see should keep first and fifth element (mainland and Corsica)
    french_primary_land = intermediate_poly.sort_values('area', ascending=False).iloc[[0, 5]]
    os.makedirs(f"{root_dir}/data/shapefiles/clean/france_primary_land", exist_ok=True)
    french_primary_land.to_file(f"{root_dir}/data/shapefiles/clean/france_primary_land/france_primary_land.shp")

    # Recover island masses
    final_land_filter = french_primary_land.overlay(eu_geoms, how='union')
    final_land_filter = final_land_filter.overlay(france_shape, how='intersection')
    os.makedirs(f"{root_dir}/data/shapefiles/clean/france_final_land_filter", exist_ok=True)
    final_land_filter.to_file(f"{root_dir}/data/shapefiles/clean/france_final_land_filter/france_final_land_filter.shp")


if __name__ == '__main__':
    main()
