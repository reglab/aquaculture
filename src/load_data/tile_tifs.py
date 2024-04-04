#
# Tile and convert French aerial Tif files into jpeg format for YOLO
#

import argparse
import os
from typing import Dict
from osgeo import gdal
from tqdm import tqdm
import glob


def split_all_tiffs(local_dir_paths: Dict[str, str], tilesize: int = 1024) -> None:
    """
    Split an entire tiff file into smaller tiff files
    
    Inputs:
        local_dir_paths: dictionary containing local paths
        tilesize:
    """

    # get list of all tifs
    initial_tifs = glob.glob(local_dir_paths['initial_tiff'] + '/*.tif')

    for t in initial_tifs:

        tif_name = os.path.basename(t)
            
        dset = gdal.Open(t)
        width = dset.RasterXSize
        height = dset.RasterYSize
        
        for i in tqdm(range(0, width, tilesize), desc='Splitting'):
            for j in range(0, height, tilesize):
                w = min(i + tilesize, width) - i
                h = min(j + tilesize, height) - j
                output_file = '{}_{}_{}.tif'.format(tif_name, i, j)
                output_path = os.path.join(local_dir_paths['tiled_tiff'], output_file)

                if os.path.exists(output_path):
                    continue

                # Create smaller tif file of subregion
                gdal.Translate(output_path, 
                            t, 
                            format="GTiff", 
                            srcWin=[i, j, w, h])


def create_jpegs(local_dir_paths: Dict[str, str]) -> None:
    '''
    Inputs:
        opt: configuration parameters
        json_metadata: dictionary containing image metadata
        gc_dir_paths: dictionary containing gcs paths
        make crop size none if not to be cropped to crop_size x crop_size
    '''

    # get list of all tiled tif files
    tiled_tifs = glob.glob(local_dir_paths['tiled_tiff'] + '/*.tif')

    # check if image folder exists yet
    if not os.path.exists(local_dir_paths['images']):
        os.makedirs(local_dir_paths['images'])     
    
    for t in tiled_tifs:
        
        tif_name = os.path.basename(t)
        jpeg_name = tif_name.replace(".tif", ".jpeg")
        
        jpeg_path = os.path.join(local_dir_paths['images'], jpeg_name)

        if not os.path.exists(jpeg_path):
            gdal.Translate(jpeg_path,t,options='-ot Byte -of JPEG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_tif_dir', type=str, help='Directory containing large tifs')
    parser.add_argument('--tiled_tif_dir', type=str, help='Directory to save tiled tifs')
    parser.add_argument('--jpeg_dir', type=str, help='Directory to save jpegs (converted from tiled tifs)')
    args = parser.parse_args()

    local_dir_paths = {'initial_tiff': args.initial_tif_dir,
                       'tiled_tiff': args.tiled_tif_dir,
                       'images': args.jpeg_dir}

    split_all_tiffs(local_dir_paths)
    create_jpegs(local_dir_paths)
