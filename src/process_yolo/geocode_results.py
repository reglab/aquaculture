#
# Initially post process French aquaculture detection results.
# This script results in a file that has all geocoded detections.
#
import argparse
import os
import glob
import cv2 as cv
import geopandas as gpd
import numpy as np
from google.cloud import storage
from google.api_core.exceptions import NotFound as google_NotFound
from shapely.geometry import box, point
from pyproj import Transformer
from tqdm import tqdm

from src.utils import (
    load_download_bboxes, CRS_DICT, IM_WIDTH, IM_HEIGHT, LARGE_TIF_SIZE, deduplicate_download_boxes,
    deduplicate_gdf_with_bboxes
)
import src.file_utils as file_utils


REVERSE_CLASS_MAPPING = {
    0: 'circle_farm',
    1: 'square_farm',
    2: 'triangle_farm',
    3: 'other_farm',
    4: 'rectangle_farm'
}
transfomer = Transformer.from_crs(CRS_DICT['mapping'], CRS_DICT['area'])


def download_positive_ims(
        label_folder, gcs_im_folder, pos_im_folder, gcs_bucket):
    '''
    download all images that have an associated label file / detection

    Inputs:
        label_folder: path to folder containing all .txt label files
        gcs_im_folder: path to folder on GCS containing all
                       images ran through the model
        pos_im_folder: path to local directory to download images to
    '''

    labels = glob.glob(label_folder + '/*.txt')

    if not os.path.exists(pos_im_folder):
        os.mkdir(pos_im_folder)

    client = storage.Client()
    bucket = client.get_bucket(gcs_bucket)

    for l in tqdm(labels):
        im_name = os.path.basename(l).replace('.txt', '.jpeg')
        gcs_im_path = os.path.join(gcs_im_folder, im_name)
        local_im_path = os.path.join(pos_im_folder, im_name)

        if not os.path.exists(local_im_path):

            blob = bucket.blob(gcs_im_path)

            try:
                blob.download_to_filename(local_im_path)
            except google_NotFound:
                print(f'[WARNING] Image not found: {local_im_path}')

    return


def convert_pix_to_m_bboxes(x, y, label, wanted_bboxes, large_tif_size=LARGE_TIF_SIZE):
    '''
    How this algorithm geocodes bboxes. Use bounding coords of jpeg
    images to geocode.

    Inputs:
        x: x pixel of point to convert
        y: y pixel of point to convert
        label: path to name of YOLO detection label
        wanted_bboxes: dataframe containing bounding boxes of large tif
                       downloads (created by download_french_data.py)
        large_tif_size: (default LARGE_TIF_SIZE 1024*6) -- size of large
                        downloaded tifs

    Output: tuple (x_m,y_m) location of wanted point (same coord
                  system as wanted_bboxes) EPSG:3857.
    '''

    _, bbox_ind, x_offset, y_offset = os.path.basename(label).replace('.txt', '').split('_')

    large_tif_bbox = wanted_bboxes.loc[int(bbox_ind), 'geometry']

    x_loc = x + int(x_offset)
    y_loc = y + int(y_offset)

    xmin_m, ymin_m, xmax_m, ymax_m = large_tif_bbox.bounds

    x_m = x_loc * ((xmax_m - xmin_m) / large_tif_size) + xmin_m  # m per pixel -- left side = lower x
    y_m = ymax_m - y_loc * ((ymax_m - ymin_m) / large_tif_size)  # m per pixel -- bottom = lower y

    return x_m, y_m


def geocode_all_detections(
    label_path: str,
    im_folder: str ,
    download_bboxes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    '''
    Geocodes all detections and creates a new file containing information
    about those detections.

    Inputs:
        label_path: path to folder containing all .txt detection files
        im_folder: directory housing images
        download_bboxes: GeoDataFrame containing bounding boxes used in the download query.
        detection_path: path to save detections file to (needs to be a .geojson file)

    Output: dataframe containing image names, bounding boxes, and geocoded boxes
    (file is also saved)
    '''

    label_paths = glob.glob(label_path + '/*.txt')

    df = {'image': [],
      'xmin': [],
      'xmax': [],
      'ymin': [],
      'ymax': [],
      'xmin_m': [],
      'xmax_m': [],
      'ymin_m': [],
      'ymax_m': [],
      'type': [],
      'year': [],
      'det_conf':[],
      'geometry':[]}

    for l in tqdm(label_paths):
        inf_bboxs = np.loadtxt(l)

        im_name = os.path.basename(l).replace('.txt', '.jpeg')
        year = int(im_name.split('_')[0][-4:])

        im_path = os.path.join(im_folder, im_name)

        # need to load image to get width and height
        im = cv.imread(im_path)

        if im is None:
            continue

        height, width, _ = im.shape

        if inf_bboxs.ndim == 1:
            inf_bboxs = [inf_bboxs]

        for bbox in inf_bboxs:

            xmin = int(IM_WIDTH*(bbox[1]-bbox[3]/2))
            ymin = int(IM_HEIGHT*(bbox[2]-bbox[4]/2))
            xmax = int(IM_WIDTH*(bbox[1]+bbox[3]/2))
            ymax = int(IM_HEIGHT*(bbox[2]+bbox[4]/2))

            detection_conf = bbox[5]

            # ymin/max for meters is swapped bc of differences in ways coords
            # work in images vs geographically. This is in EPSG:3857
            xmin_m, ymax_m = convert_pix_to_m_bboxes(xmin, ymin, l, download_bboxes)
            xmax_m, ymin_m = convert_pix_to_m_bboxes(xmax, ymax, l, download_bboxes)

            farm_type = REVERSE_CLASS_MAPPING[bbox[0]]

            det_box_3857 = box(xmin_m, ymin_m, xmax_m, ymax_m)

            # Convert to EPSG:3035 for area measurements
            xmin_a, ymax_a = transfomer.transform(xmin_m, ymax_m)
            xmax_a, ymin_a = transfomer.transform(xmax_m, ymin_m)

            df['image'].append(im_name)
            df['xmin'].append(xmin)
            df['xmax'].append(xmax)
            df['ymin'].append(ymin)
            df['ymax'].append(ymax)
            df['xmin_m'].append(xmin_a)  # EPSG:3035
            df['xmax_m'].append(xmax_a)  # EPSG:3035
            df['ymin_m'].append(ymin_a)  # EPSG:3035
            df['ymax_m'].append(ymax_a)  # EPSG:3035
            df['type'].append(farm_type)
            df['year'].append(year)
            df['det_conf'].append(detection_conf)
            df['geometry'].append(det_box_3857)  # EPSG:3857

    df = gpd.GeoDataFrame(df, crs=f"EPSG:{CRS_DICT['mapping']}")  # We keep the mapping CRS but use area measurements from EPSG:3035
    df = df.to_crs(4326)

    return df


def remove_land_detections(
    detections: gpd.GeoDataFrame, path_to_shapefile: str
) -> gpd.GeoDataFrame:
    """ Removes probable land detections from detections.

    Args:
        detections: a GeoDataFrame of detections.
        path_to_shapefile: path to land shapefile

    Returns:
        ocean-based detections.
    """

    french_land = gpd.read_file(path_to_shapefile)
    french_land.to_crs(detections.crs, inplace=True)
    land_detections = detections.sjoin(french_land, how='inner')
    ocean_detections = detections[~detections.index.isin(land_detections.index)]

    return ocean_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_bucket', type=str, help='Name of bucket in GCS')
    parser.add_argument('--gcs_im_path', type=str, help='Directory in GCS bucket where images are stored')
    parser.add_argument(
        '--detection_txt', type=str,
        help='Path to folder containing all .txt detection files produced by running inference')
    args = parser.parse_args()

    # Root directory
    root_dir = file_utils.get_root_path()

    # path to local directory to download images with detections to
    im_path = f'{root_dir}/data/french_inference_images'

    # path to file wanted_bboxes.csv created by download_french_data.py
    download_bboxes_path = f'{root_dir}/data/wanted_bboxes.csv'

    # path to land shapefile
    path_to_land_shapefile = f"{root_dir}/data/shapefiles/clean/france_final_land_filter/france_final_land_filter.shp"

    # path to save detections file to (needs to be a .geojson file)
    detection_path = f'{root_dir}/output/detections.geojson'

    # path to save detections screened for being in the ocean to (needs to be a .geojson file)
    ocean_detection_path = f'{root_dir}/output/ocean_detections.geojson'

    print('DOWNLOADING IMAGES...')
    download_positive_ims(args.detection_txt, args.gcs_im_path, im_path, args.gcs_bucket)
    print('DOWNLOADED')

    print('CREATING DETECTIONS.GEOJSON...')
    download_bboxes = load_download_bboxes(download_bboxes_path)
    detections = geocode_all_detections(args.detection_txt, im_path, download_bboxes)

    print('DEDUPLICATING DETECTIONS WHERE IMAGES OVERLAP...')
    dedup_boxes = deduplicate_download_boxes(
        download_bboxes, path=f"{root_dir}/data/wanted_bboxes_dedup.csv")
    detections['bbox_ind'] = detections['image'].apply(lambda f: f.split('_')[1])
    detections.to_crs(dedup_boxes.crs, inplace=True)
    detections = deduplicate_gdf_with_bboxes(dedup_boxes=dedup_boxes, gdf_todedup=detections)
    detections.drop('bbox_ind', axis=1, inplace=True)
    
    print('SAVING DETECTIONS...')
    detections.to_file(detection_path)

    # Remove land detections
    print('REMOVING PROBABLE LAND DETECTIONS...')
    ocean_detections = remove_land_detections(detections, path_to_land_shapefile)
    print('SAVING OCEAN DETECTIONS...')
    ocean_detections.to_file(ocean_detection_path, index=True)
    print('Done.')
