#
# Loads the france_maritime_shapefile and world_shapefiles used to download the French aerial data.
#

import io
import os
import geopandas as gpd
import requests
import zipfile

import src.file_utils as file_utils


def load_data() -> None:
    """
    Saves the European coastline and French territory shapefiles locally.
    """
    root_dir = file_utils.get_root_path()
    output_dir = f'{root_dir}/data/shapefiles'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # France maritime shapefile
    link = 'https://geo.vliz.be/geoserver/wfs?request=getfeature&service=wfs&version=1.0.0&typename=MarineRegions:eez_iho&outputformat=SHAPE-ZIP&filter=%3COr%3E%3COr%3E%3CPropertyIsEqualTo%3E%3CPropertyName%3Emrgid%3C%2FPropertyName%3E%3CLiteral%3E25185%3C%2FLiteral%3E%3C%2FPropertyIsEqualTo%3E%3CPropertyIsEqualTo%3E%3CPropertyName%3Emrgid%3C%2FPropertyName%3E%3CLiteral%3E25609%3C%2FLiteral%3E%3C%2FPropertyIsEqualTo%3E%3C%2FOr%3E%3CPropertyIsEqualTo%3E%3CPropertyName%3Emrgid%3C%2FPropertyName%3E%3CLiteral%3E25612%3C%2FLiteral%3E%3C%2FPropertyIsEqualTo%3E%3C%2FOr%3E'
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.join(output_dir, 'raw', 'france_maritime_shapefile'))
    # Final file path should be data/shapefiles/raw/france_maritime_shapefile/eez_iho.shp

    # World shorelines shapefile
    link = 'https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip'
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.join(output_dir, 'raw', 'world_shapefiles'))
    # Final file path is data/shapefiles/raw/world_shapefiles/GSHHS_shp/i/GSHHS_i_L1.shp


if __name__ == '__main__':
    load_data()
