''' Create list of potential facilities from processed detections '''

import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, MultiPolygon

from src.utils import CRS_DICT


def get_clusters(center_points, distance_threshold, min_samples):

    db = DBSCAN(eps=distance_threshold, min_samples=min_samples).fit(center_points)
    
    return db.labels_

def DBSCAN_cluster(
    cages,
    facilities_path,
    cluster_variable,
    distance_threshold=10,
    amnt_min_clusters=5,
    include_area=True,
    save=True,
    return_detections=False
):
    '''
    cluster groups of detections using sklearn's DBSCAN algorithm
    this algorithm also filters out noisy detections

    
    Inputs:
        detections: geocoded detections (created by postprocess_results.py)
        facilities_path: path to save grouped facilities
        conf_thresh: float b/t 0.0-1.0, confidence threshold to screen for
        distance_threshold: meter distance to use between cages to consider
                            them part of the same facility (default 10m)
        amnt_min_clusters: minimum amount of cages required to consider a group
                           of detections a facility (default 5)
        include_area: whether or not to include area calculations from calc_net_areas.py
        cluster_variable: whether to cluster at the year or period level

    Output: geodataframe containing facility information (also saved to file) (all geometries in EPSG:3857)
    '''
    # Check that we have a unique identifying cage ID that we can use to link back once we have the
    # facility-level data
    assert 'index' in cages.columns and len(cages['index'].unique()) == len(cages), '[ERROR] Check cage ID'

    # Check farm type names
    if len(cages) > 0:
        assert 'circle_farm' in cages['type'].unique() or 'square_farm' in cages['type'].unique()

    # Convert CRS (so that distance thresholds are correctly computed)
    assert cages.crs == f'EPSG:{CRS_DICT["area"]}', '[ERROR] Check CRS'

    # what we want our results data to look like
    df = {
        'num_square_farms': [],
        'num_circle_farms': [],
        'num_rectangle_farms': [],
        cluster_variable: [],
        'noise_points': [],
        'square_farm_geoms': [],
        'circle_farm_geoms': [],
        'rectangle_farm_geoms': [],
        'cage_ids': []
    }
    
    # get all indices for detections within facilities
    all_farm_detections_indices = []

    geometries = []

    if include_area:
        areas = []
        area_vars = []
        min_areas = []
        max_areas = []

    # Check cluster variable
    if cluster_variable not in cages.columns:
        print('[ERROR] Check clustering variable')

    # then need to do by cluster_variable
    for y in cages[cluster_variable].unique():
        cages_y = cages[cages[cluster_variable] == y].copy()


        # get center points of detections in both m and lat/lon
        center_points = []
        for _ ,row in cages_y.iterrows():
            cp = row['geometry'].centroid.xy
            center_points.append([cp[0][0],cp[1][0]])

        center_points = np.array(center_points)

        labels = get_clusters(center_points, distance_threshold, amnt_min_clusters)

        # Number of clusters in labels, ignoring noise if present.
        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        #print('year: %s' %y)
        #print("Estimated number of clusters: %d" % n_clusters_)
        #print("Estimated number of noise points: %d" % n_noise_)

        for l in np.unique(labels):
            if l == -1:
                continue
            points_of_cluster = center_points[labels==l,:]
            centroid_of_cluster = np.mean(points_of_cluster, axis=0)

            farm_indices = np.where(labels == l)[0]
            farm_detections = cages_y.iloc[farm_indices].copy()

            # Convert to 3857 so that we append the cage geometries in this CRS
            farm_detections.to_crs(f'EPSG:{CRS_DICT["mapping"]}', inplace=True)

            df['num_circle_farms'].append(sum(farm_detections['type'] == 'circle_farm'))
            df['num_square_farms'].append(sum(farm_detections['type'] == 'square_farm'))
            df['num_rectangle_farms'].append(sum(farm_detections['type'] == 'rectangle_farm'))
            df['cage_ids'].append(farm_detections['index'].tolist())
            
            # Add farm geometries
            for ftype in ['circle', 'square', 'rectangle']:
                df[ftype + '_farm_geoms'].append(
                    MultiPolygon(farm_detections.loc[farm_detections['type'] == ftype + "_farm", 'geometry'].tolist()
                    )
                )
            
            all_farm_detections_indices += farm_detections.index.tolist()

            if include_area:
                areas.append(farm_detections['area'].sum())
                area_vars.append(float(np.sum(farm_detections['area_var'])))
                min_areas.append(farm_detections['min_area'].sum())
                max_areas.append(farm_detections['max_area'].sum())

            # need to deal w year
            df[cluster_variable].append(y)
            df['noise_points'].append(n_noise_)

            geometries.append(Point(centroid_of_cluster[0],centroid_of_cluster[1]))
    
    if include_area:
        df['area'] = areas
        df['area_var'] = area_vars
        df['min_area'] = min_areas
        df['max_area'] = max_areas

    df = gpd.GeoDataFrame(df, geometry=geometries, crs=cages.crs)

    # Set up a facility ID
    df.reset_index(inplace=True, drop=True)
    df['facility_index'] = df.index

    # Convert the facility Point geometries to EPSG:3857
    df.to_crs(f'EPSG:{CRS_DICT["mapping"]}', inplace=True)

    if save:
        # Cast farm geometries to WKT
        for farm_col in df.columns[df.columns.str.endswith('_farm_geoms')]:
            df[farm_col] = df[farm_col].apply(lambda x: x.wkt)

        df.to_file(facilities_path, driver='GeoJSON')

    # get all detections within any facility and convert back to EPSG:3857
    facility_detections = cages.loc[np.unique(all_farm_detections_indices)]
    facility_detections_path = facilities_path.replace('.geojson','_detections.geojson')
    facility_detections.to_crs(f'EPSG:{CRS_DICT["mapping"]}', inplace=True)

    if save:
        facility_detections.to_file(facility_detections_path, driver='GeoJSON')

    if return_detections:
        return facility_detections
    else:
        return df
    

def predictions_cluster(
    predictions,
    facilities_path,
    cluster_variable,
    conf_thresh=0.5,
    distance_threshold=10,
    amnt_min_clusters=5,
    include_area=True,
    save=True,
    return_detections=False
):
    
    # next -- convert to meters
    preds = predictions.copy()
    preds = preds[preds['det_conf'] >= conf_thresh]

    result = DBSCAN_cluster(
        cages=preds,
        facilities_path=facilities_path,
        distance_threshold=distance_threshold,
        amnt_min_clusters=amnt_min_clusters,
        include_area=include_area,
        save=save,
        return_detections=return_detections,
        cluster_variable=cluster_variable
    )

    return result
