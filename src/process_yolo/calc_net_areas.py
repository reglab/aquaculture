#
# Estimate the sqm surface area of detected cages
#

import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import geopandas as gpd

import src.file_utils as file_utils


REVERSE_CLASS_MAPPING = {
    0: 'circle_farm',
    1: 'square_farm',
    2: 'triangle_farm',
    3: 'other_farm',
}

LARGE_TIF_SIZE = 1024*6


def get_circle_area_from_bbox(bbox_width, bbox_height, x_border=False, y_border=False):
    '''
    Estimate area of circle from its bounding box

    Inputs:
        bbox_width: float, in meters
        bbox_height: float, in meters
        x_border: boolean, whether or not the bbox is on the left/right image border
        y_border: boolean, whether or not the bbox is on the top/bottom image border

    Output: (estimated area in sqm, variance in sqm, min area, max area)
    '''

    # Use area of an ellipse equation
    if x_border or y_border:
        if x_border:
            if y_border:
                min_area = bbox_height * bbox_width / 2  # Treat as approximating triangle
                max_area = np.pi * bbox_height * bbox_width / 4  # Treat as quarter ellipse.
            else:
                min_area = bbox_height * bbox_width / 2  # Treat as approximating triangle
                max_area = np.pi * (bbox_height / 2) * bbox_width / 2  # Treat as half ellipse
        elif y_border:
            min_area = bbox_height * bbox_width / 2  # Treat as approximating triangle
            max_area = np.pi * bbox_height * (bbox_width / 2) / 2  # Treat as half ellipse
    
        estimate = (min_area + max_area) / 2
        var = ((max_area - min_area)**2) / 12
        
        return estimate, var, min_area, max_area
    else:
        a = bbox_width / 2
        b = bbox_height / 2

        return np.pi * a * b, 0, np.pi * a * b, np.pi * a * b


def get_square_area_from_bbox(bbox_width, bbox_height):
    '''
    Estimate area of square from its bounding box.

    Note: we assume that the cage's orientation within the bounding box is uniformly distributed.

    Inputs:
        bbox_width: float, in meters
        bbox_height: float,  in meters

    Output: tuple of (central estimate, variance, min area, max area) in sqm
    '''

    # Since we assume the cage's orientation within the bounding box is uniformly distributed,
    # the central area estimate is (lower + upper) / 2
    min_area = bbox_width*bbox_height*(1/2)
    max_area = bbox_width*bbox_height

    estimate = (min_area + max_area) / 2
    var = ((max_area - min_area)**2) / 12 

    return estimate, var, min_area, max_area


def calc_all_areas(detections_path,im_folder):
    '''
    Calculates area (in sqm) of all detections and adds that info to
    detections.geojson file

    Inputs:
        detections_path: path to file containing geocoded detections (created
                         by geocode_results.py)
        im_folder: directory housing images

    Output: detections geodataframe now containing sqm area estimations
            (also resaved to detections.geojson file)
    '''

    # load the geocoded data
    detections = gpd.read_file(detections_path)

    # Note: CRS conversion to meters is unnecessary since the area calcs are performed directly on the
    # width and height per the bounding box coordinates in EPSG:3035
    detections_m = detections.copy()

    # for each detection calculate approximate area
    areas = []
    area_vars = []
    min_areas = []
    max_areas = []
    for _, row in tqdm(detections_m.iterrows(), total=len(detections_m)):

        # load image first
        im_path = os.path.join(im_folder, row['image'])
        im = cv.imread(im_path)
        
        if im is None:
            continue

        height, width, _ = im.shape

        bbox_width_m = row['xmax_m'] - row['xmin_m']
        bbox_height_m = row['ymax_m'] - row['ymin_m']

        if row['type'] == 'circle_farm':
            # check if detection is on the border
            x_border = False
            y_border = False
            if row['xmin'] == 0 or row['xmax'] == width:
                x_border = True

            if row['ymin'] == 0 or row['ymax'] == height:
                y_border = True 

            area, area_var, min_area, max_area = get_circle_area_from_bbox(
                bbox_width_m, bbox_height_m, x_border=x_border, y_border=y_border)
                
        elif row['type'] == 'square_farm':
            area, area_var, min_area, max_area = get_square_area_from_bbox(bbox_width_m, bbox_height_m)

        areas.append(area)
        area_vars.append(area_var)
        min_areas.append(min_area)
        max_areas.append(max_area)

    detections['area'] = areas
    detections['area_var'] = area_vars
    detections['min_area'] = min_areas
    detections['max_area'] = max_areas

    detections.to_file(detections_path, driver='GeoJSON')

    return detections


if __name__ == '__main__':

    root_dir = file_utils.get_root_path()

    # path to detections file (created by geocode_results.py)
    detections_path = f'{root_dir}/output/detections.geojson'

    # path to ocean screened detections file (created by geocode_results.py)
    ocean_detections_path = f'{root_dir}/output/ocean_detections.geojson'

    # path to local directory containing images with positive detections
    im_path = f'{root_dir}/data/french_inference_images'

    print('CALCULATING ALL AREAS FOR OCEAN DETECTIONS...')
    calc_all_areas(ocean_detections_path, im_path)
    print('DONE')

    print('CALCULATING ALL AREAS FOR ALL DETECTIONS...')
    calc_all_areas(detections_path, im_path)
    print('DONE')
