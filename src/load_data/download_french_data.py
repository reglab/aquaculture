import argparse
import requests.exceptions
from owslib.wms import WebMapService as wms
from shapely.geometry import box
import numpy as np
import os
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from google.cloud import storage

from src.utils import is_blank
import src.file_utils as file_utils


def create_wms_obj():
    '''
    create wms object for french downloading
    (contains a hardcoded link to French orthoimagery data we use)

    Returns: wms_obj
    '''

    wms_url = 'https://wxs.ign.fr/orthohisto/geoportail/r/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities'
    wms_obj = wms(wms_url, version='1.1.1')

    return wms_obj


def download_tiff(wms_obj, wms_layer, region_bbox, bucket, out_path):
    '''
    download tiff from wms object

    Inputs:
        wms_obj: wms object, created by create_wms_obj
        wms_layer: layer to download (of the format 'ORTHOIMAGERY.ORTHOPHOTOS{YYYY}')
                   where YYYY is the year of orthoimagery desired
        region_bbox: bounding box of desired geography
        bucket: GCS bucket to download imagery to
        out_path: path to download to on GCS bucket
    '''

    blob = bucket.blob(out_path)

    if blob.exists():
        print('exists')
        return

    # first check with low resolution if anything exists
    img = wms_obj.getmap(layers=[wms_layer],
                    styles=['normal'],
                    srs='EPSG:3857',
                    bbox=(region_bbox),
                    size=(1024, 1024),
                    format='image/tiff',
                    transparent=True)

    # check if blank and don't download if so
    if is_blank(img_bytes=img.read()):
        print('blank image')
        return
    else:

        blob = bucket.blob(out_path)

        if blob.exists():
            return
        print(f'Loading: {out_path}')
        # download with higher resolution
        img = wms_obj.getmap(layers=[wms_layer],
                styles=['normal'],
                srs='EPSG:3857',
                bbox=(region_bbox),
                size=(1024*6, 1024*6),
                format='image/tiff',
                transparent=True)

        blob.upload_from_string(img.read(), content_type='image/tiff')


def create_region_bbox_from_latlon(lat, lon, meter_buff=200):
    '''
    creates a bounding box of lat/longitude points of a wanted meter size
    in this case, 200m x 200m for 1046 x 1046 pixels

    Inputs:
        lat: central lattitude of wanted point
        lon: central longitude of wanted point
        meter_buff: size (in m) of wanted region, default 200

    Returns: tuple of (min_lon, min_lat, max_lon, max_lat)
    '''

    # need to convert lat / lon to projected CRS
    conv_lat = lat
    conv_lon = lon

    min_lat = conv_lat - meter_buff/2
    max_lat = conv_lat + meter_buff/2

    min_lon = conv_lon - meter_buff/2
    max_lon = conv_lon + meter_buff/2

    region = (min_lon, min_lat, max_lon, max_lat)

    return region


def tile_shapefile(coast_shape, size=200):
    '''
    tiles a shapefile into wanted 200x200 boxes

    Inputs:
        coast_shape: geopandas GeoDataFrame containing multipolygons of the
                   French coast with a buffer
        size: size (in meters) of tiles, default 200m

    Returns: list of tiled bounding boxes within the shapefile
    '''

    print('using size %s' %size)
    wanted_bboxes = []
    for multipolygon in coast_shape:

        if multipolygon.area == 0:
            continue

        bounds = multipolygon.bounds

        for j in tqdm(np.arange(bounds[0],bounds[2]+size, size)):
            for k in np.arange(bounds[1],bounds[3]+size, size):

                bounding_box = box(j, k, j+size, k+size)

                if multipolygon.intersects(bounding_box):

                    wanted_bboxes.append(bounding_box)
    
    return wanted_bboxes


def download_ims_within_shapefile(wms_layer, coast_shape, bucket, out_dir, gcs_bucket_name):
    '''
    download all images that intersect with shapefile

    Inputs:
        wms_layer: layer to download (of the format 'ORTHOIMAGERY.ORTHOPHOTOS{YYYY}')
                   where YYYY is the year of orthoimagery desired
        coast_shape: geopandas GeoDataFrame containing multipolygons of the
                   French coast with a buffer
        bucket: GCS bucket to download imagery to
        out_dir: directory on GCS bucket to download imagery to
        gcs_bucket_name: name of the GCS bucket
    '''

    print('ACCESSING WMS OBJECT')
    wms_obj = create_wms_obj()
    coverage_box = wms_obj[wms_layer].boundingBox
    coverage_box = box(coverage_box[0],coverage_box[1],coverage_box[2],coverage_box[3])
    print('FOUND OBJECT WITH COVERAGE %s' %coverage_box)

    bboxes_path = os.path.join(out_dir, 'wanted_bboxes.csv')

    wanted_bboxes_gcs_path = os.path.join(f'gs://{gcs_bucket_name}/', bboxes_path)
    bboxes_blob = bucket.blob(bboxes_path)

    if bboxes_blob.exists():
        print('FOUND A WANTED BBOXES CSV...')
        wanted_bboxes = pd.read_csv(wanted_bboxes_gcs_path)
        wanted_bboxes = list(gpd.GeoSeries.from_wkt(wanted_bboxes['geometry']))
        print('LOADED %s WANTED IMAGES' %len(wanted_bboxes))
    else:
        print('OBTAINING WANTED BBOXES...')
        wanted_bboxes = tile_shapefile(coast_shape,size=200*6)
        print('FOUND %s WANTED IMAGES' %len(wanted_bboxes))
        print('SAVING TO CSV FILE...')
        wanted_bboxes_df = gpd.GeoDataFrame(geometry=wanted_bboxes)
        wanted_bboxes_df.to_csv(wanted_bboxes_gcs_path)
        print('SAVED')

    print('DOWNLOADING TIFS...')
    for i, bbox in tqdm(enumerate(wanted_bboxes)):
        out_name = wms_layer + '_' + str(i) + '.tif'
        out_path = os.path.join(out_dir, out_name)
        try:
            download_tiff(wms_obj, wms_layer, bbox.bounds, bucket, out_path)
        except requests.exceptions.ConnectionError:
            print(f'Connection error: {i}')
            continue
    print('FINISHED DOWNLOADING')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_bucket', type=str, help='Name of bucket in GCS')
    parser.add_argument('--gcs_path', type=str, help='Directory in GCS bucket where data will be downloaded')
    args = parser.parse_args()

    # root dir
    root_dir = file_utils.get_root_path()

    # path to maritime shapefile
    maritime_france_shp_path = os.path.join(root_dir, 'data/shapefiles/raw/france_maritime_shapefile/eez_iho.shp')

    # path to world shorelines shapefile
    world_shorelines_path = os.path.join(root_dir, 'data/shapefiles/raw/world_shapefiles/GSHHS_shp/i/GSHHS_i_L1.shp')

    # wanted layers to download in wms object
    wanted_layers = [f'ORTHOIMAGERY.ORTHOPHOTOS{y}' for y in range(2000, 2021)] + ['ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2021']

    # Set up Google Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket(args.gcs_bucket)

    # load maritime shapefile
    france_maritime = gpd.read_file(maritime_france_shp_path)
    france_maritime = france_maritime.to_crs(3857)

    # load world shorelines shapefile
    world_shorelines = gpd.read_file(world_shorelines_path)
    world_shorelines = world_shorelines.to_crs(3857)

    # buffer world shorelines by 2km
    buffered_shorelines = world_shorelines.buffer(2000)

    # find intersection between mediterranean france territory & close shoreline
    med_coastal = buffered_shorelines.intersection(france_maritime['geometry'][2])  # mediterranean polygon

    for wanted_layer in wanted_layers:
        download_ims_within_shapefile(wanted_layer, med_coastal, bucket, args.gcs_path, args.gcs_bucket)
