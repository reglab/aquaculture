"""
Downloads the bathymetry data from European Marine Observation and Data
Network (https://emodnet.ec.europa.eu/geonetwork/srv/eng/catalog.search#/metadata/ff3aff8a-cff1-44a3-a2c8-1910bf109f85)
"""
import os
import io
import rasterio
import requests
import zipfile
from rasterio.merge import merge as r_merge

import src.file_utils as file_utils

if __name__ == '__main__':
    # Output path
    root_dir = file_utils.get_root_path()
    output_dir = f'{root_dir}/data/bathymetry'
    os.makedirs(output_dir, exist_ok=True)

    year = 2022
    tiles = ['F4', 'F5', 'E5']

    # ESRI
    base_link = 'https://downloads.emodnet-bathymetry.eu/v11/{}_{}.asc.zip'

    datasets = []
    for tile in tiles:
        print(f'[INFO] Loading {tile}')

        # Download zip file
        link = base_link.format(tile, year)
        r = requests.get(link)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(output_dir)
        file_name = '{}_{}_mean'
        file_name = file_name.format(tile, year)
        src = rasterio.open(f'{output_dir}/{file_name}.asc')
        datasets.append(src)

    # Merge and save
    r_merge(datasets=datasets, dst_path=f'{output_dir}/EMOD_{year}.tif')

