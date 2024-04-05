import itertools
import os
from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Any, List
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from src.cluster_facilities import predictions_cluster
import src.get_kfold_cluster_performance_cfg as cfg
from src.utils import (
    load_download_bboxes, load_cf_labels, load_cf_images, load_Trujillo_locations_deduped, mark_land_images,
    deduplicate_gdf_with_bboxes, deduplicate_download_boxes, get_french_image_boxes,
    generate_image_file_name_str, generate_image_specs_from_file_name
)
import src.file_utils as file_utils


CONF_BINS = [0, 0.3, 0.5, 0.8, 1]


def load_datasets_for_model_evaluation(
        detections_path: str,
        landshape_path: str,
        base_dir: str) -> Dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Loads the labels, detections (ocean, all), and other datasets used for stratification and model evaluation.
    :param detections_path: Path to the detections.geojson
    :param landshape_path: Path to the French land filter
    :param base_dir:
    :return: Dictionary including each of the datasets
    """
    # Load download bboxes and image metadata
    print('Loading images.')
    download_bboxes = load_download_bboxes(f"{base_dir}/data/wanted_bboxes.csv")
    dedup_boxes = deduplicate_download_boxes(download_bboxes, path=f"{base_dir}/data/wanted_bboxes_dedup.csv")
    all_images = get_french_image_boxes(
        download_bboxes=download_bboxes, path_to_image_gdf=f'{base_dir}/data/image_boxes_years.geojson')
    all_images['image'] = all_images.apply(lambda row: generate_image_file_name_str(d=row), axis=1)

    # Load labels
    print('Loading labels.')
    labels = load_cf_labels().rename(columns={"fn": "image"})
    labels = labels[labels['type'].isin(['circle_cage', 'square_cage'])]
    labels['type'] = labels['type'].replace({'circle_cage': 'circle_farm', 'square_cage': 'square_farm'})
    labels = deduplicate_gdf_with_bboxes(dedup_boxes=dedup_boxes, gdf_todedup=labels)

    # Load detections in labeled areas
    print('Loading predictions.')
    detections = gpd.read_file(detections_path)
    detections = detections[detections['type'].isin(['circle_farm', 'square_farm'])]
    detections['bbox_ind'] = detections['image'].apply(lambda f: generate_image_specs_from_file_name(f)['bbox_ind'])
    detections = deduplicate_gdf_with_bboxes(dedup_boxes=dedup_boxes, gdf_todedup=detections, path=None)

    # Load Trujillo et al. locations
    # Note that we use the EPSG:3857 here to define the 1km boxes since this was used to originally design the
    # strata. 
    print('Loading Trujillo et al. locations.')
    jennifer_locations = load_Trujillo_locations_deduped(f"{base_dir}/data/aquaculture_med_dedupe.csv")
    jennifer_locations['1km_box'] = jennifer_locations['geometry'].apply(
        lambda p: box(p.x - 1000, p.y - 1000, p.x + 1000, p.y + 1000)
    )

    # Load land geometry for denoting land images
    print('Load France shapefile.')
    france_land = gpd.read_file(landshape_path)

    # Load sampled images
    print('Load sampled images.')
    sampled_images = load_cf_images()
    sampled_images['year'] = sampled_images['image'].apply(lambda f: generate_image_specs_from_file_name(f)['year'])
    sampled_images['bbox_ind'] = sampled_images['image'].apply(lambda f: generate_image_specs_from_file_name(f)['bbox_ind'])

    # Demarcate land images and detections on land
    all_images['only_land'] = mark_land_images(all_images, france_land)
    detections['surely_land'] = detections['image'].isin(all_images[all_images['only_land']]['image'])

    # Mark land images as sampled
    sampled_images = pd.concat([
        sampled_images,
        all_images[all_images['only_land']].drop(columns=['geometry'])
    ], axis=0, ignore_index=True)
    sampled_images['only_land'] = sampled_images['only_land'].fillna(False)

    # Set image metadata
    print('Set bucket information.')
    all_images['in_sample'] = all_images['image'].isin(sampled_images['image'])
    all_images = set_image_stats(all_images, detections, labels)
    all_images = set_buckets(all_images, jennifer_locations)
    sampled_images['bucket'] = sampled_images['image'].map(all_images.set_index('image')['bucket'])
    detections['bucket'] = detections['image'].map(all_images.set_index('image')['bucket'])
    labels['bucket'] = labels['image'].map(all_images.set_index('image')['bucket'])

    # Add a cage index
    detections.reset_index(inplace=True, drop=True)
    detections['index'] = detections.index

    # Get ocean images and detections
    ocean_images = all_images[~all_images['only_land']]
    ocean_detections = detections[~detections['surely_land']]

    # Dataset dictionary
    datasets = {
        'all_images': all_images,
        'ocean_detections': ocean_detections,
        'detections': detections,
        'ocean_images': ocean_images,
        'sampled_images': sampled_images,
        'labels': labels
    }
    return datasets


def get_tp(query: gpd.GeoDataFrame, key: gpd.GeoDataFrame) -> pd.Series:
    """Get whether query is true positive.
    
    A true positive is defined as a query that intersects with a key of the same type and year.
    
    Args:
        query: GeoDataFrame of query objects (labels or predictions).
        key: GeoDataFrame of key objects (labels or predictions).
    Returns:
        Series of booleans indicating whether query is true positive.
    """
    assert query.crs == key.crs
    joined = query.sjoin(
        key, how='left', predicate='intersects', lsuffix='query', rsuffix='key'
    )

    joined['tp'] = joined.apply(
        lambda r: True
        if r['index_key'] and r['year_query'] == r['year_key'] and r['type_query'] == r['type_key'] else False,
        axis=1
    )

    return joined.groupby(level=0)['tp'].any()


def set_image_stats(
    images: gpd.GeoDataFrame,
    detections: gpd.GeoDataFrame,
    labels: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Add detection and label information to image Dataframe.
    
    Args:
        images: a gdf of image metadata
        detecctions: a gdf of model detections
        labels: a gdf of cage labels.
        
    Returns:
        A gdf of image metadata with prediction and label information.
    """
    with_detections = images.sjoin(detections.to_crs(images.crs), how='left', predicate='intersects')
    with_detections = with_detections[
        (with_detections['image_left'] == with_detections['image_right']) | (with_detections['image_right'].isna())
    ]
    with_labels = images.sjoin(labels.to_crs(images.crs), how='left', predicate='intersects')
    with_labels = with_labels[
        (with_labels['image_left'] == with_labels['image_right']) | (with_labels['image_right'].isna())
    ]

    images_stats = images.join(
        with_detections.groupby(level=0).agg({
            'det_conf': max,
            'index_right': lambda x: len(x) if not any(pd.isna(x)) else 0
        })
    ).rename({'index_right': 'num_detections'}, axis=1).join(
        with_labels.groupby(level=0).agg({
            'index_right': lambda x: len(x) if not any(pd.isna(x)) else 0
        })
    ).rename({'index_right': 'num_labels_sample'}, axis=1)
    images_stats.loc[~images_stats['in_sample'], 'num_labels_sample'] = np.nan

    return images_stats


def set_buckets(
    ims: gpd.GeoDataFrame,
    Trujillo_locations: gpd.GeoDataFrame,
    conf_bins: List = CONF_BINS
) -> gpd.GeoDataFrame:
    """Add buckets (strata) to image metadata.
    
    Args:
        ims: image metadata
        Trujillo_locations: a gdf of Trujillo et al. locations
        conf_bins: threshold to use to create confidence bin based buckets.

    Returns:
        Image metadata with bucket information
    """
    images = ims.copy()
    images['in_jennifer_area'] = images.index.isin(
        images.sjoin(Trujillo_locations.set_geometry('1km_box'), how='inner', predicate='intersects').index.unique()
    )

    images['conf_bucket'] = pd.cut(images['det_conf'], bins=conf_bins).values.add_categories('No detection')
    images['conf_bucket'] = images['conf_bucket'].fillna('No detection')

    images.loc[images['only_land'], 'bucket'] = 'land'
    images.loc[~images['only_land'], 'bucket'] = images.loc[~images['only_land'], 'conf_bucket']
    images.loc[
        (images['in_jennifer_area']) & (~images['only_land']) & (images['bucket'] == 'No detection'), 'bucket'
    ] = 'No detection, in jennifer area'
    images.loc[
        (~images['in_jennifer_area']) & (~images['only_land']) & (images['bucket'] == 'No detection'), 'bucket'
    ] = 'No detection, outside jennifer area'

    # Convert to categorical
    images['bucket'] = pd.Categorical(images['bucket'])

    return images


def get_bucket_info_table(images: gpd.GeoDataFrame) -> pd.DataFrame:
    """Returns a table with metadata for each image bucket (stratum)
    
    Args:
        images: Image metadata, including information on the bucket each image belongs to.
        
    Returns:
        A DataFrame of bucket metadata.
    """
    # Define lambdas for aggregation.
    num_in_sample = lambda x: len(x[images.loc[x.index, 'in_sample']])
    sum_in_sample = lambda x: x[images.loc[x.index, 'in_sample']].sum()

    # Aggregate to obtain bucket metadata.
    bucket_info = images.groupby('bucket').agg({
        'num_detections': [sum, sum_in_sample],
        'image': ['size', num_in_sample],
        'num_labels_sample': sum
    }).rename({'image': 'num_images'}, axis=1)

    # Unstack aggregation functions.
    bucket_info = bucket_info.rename(
        columns={bucket_info.columns.get_level_values(1)[idx]: ['bucket', 'sample'][idx%2] for idx in range(len(bucket_info.columns))},
        level=1
    )
    
    # Clean column names
    bucket_info.columns = bucket_info.columns.map('_'.join).str.strip('_')
    bucket_info.rename({'num_labels_sample_bucket': 'num_labels_sample'}, axis=1, inplace=True)
    bucket_info['estimated_num_labels_bucket'] = (bucket_info['num_labels_sample'] / bucket_info['num_images_sample']) * bucket_info['num_images_bucket']

    return bucket_info


def get_stats_total(
        labels: gpd.GeoDataFrame,
        preds: gpd.GeoDataFrame,
) -> dict:
    """Estimate precision and recall for the population of detections and labels.
    Note: This calculation assumes 1) all predictions have been labeled; 2) there are no labels in the
    no land, no prediction, not near known facility stratum.

    Args:
        labels: labels
        preds: predictions

    Returns:
        A dictionary of performance metrics.
    """
    preds['tp'] = get_tp(preds, labels)
    precision = preds['tp'].mean()

    # * Recall
    labels['tp'] = get_tp(labels, preds)
    recall = labels['tp'].mean()

    return {'precision': precision, 'recall': recall}


def get_fold_performance(
    fold_index: Tuple[np.array, np.array],
    images: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    labels: gpd.GeoDataFrame,
    confidence_thresholds: Iterable = cfg.confidence_thresholds,
    distance_thresholds: Iterable = cfg.distance_thresholds,
    minimum_cluster_sizes: Iterable = cfg.minimum_cluster_sizes
) -> list[dict]:
    """Gets performance across a fold.
    
    Args:
        fold_index: integer indices for the fold's train/test split.
        images: images from which to create train/test split.
        predictions: model predictions
        labels: true cage bounding boxes
        confidence_thresholds: set of confidence thresholds to test.
        distance_thresholds: set of distance threshold to test.
        minimum_cluster_sizes: set of minimum cluster sizes to test
        
    Returns:
        A tuple of dictionaries with the best performance 
            on the train split using the product of precision/recall or f-score as the decision metric.
    """
    # Create splits.
    train_images, test_images = images.iloc[fold_index[0]], images.iloc[fold_index[1]]

    train_preds = predictions[predictions['image'].isin(train_images['image'])].copy()
    test_preds = predictions[predictions['image'].isin(test_images['image'])].copy()
    train_labels = labels[labels['image'].isin(train_images['image'])].copy()
    test_labels = labels[labels['image'].isin(test_images['image'])].copy()

    # Evaluate performance over hyperparameter grid.
    train_stats_dicts = []
    param_combinations = product(confidence_thresholds, distance_thresholds, minimum_cluster_sizes)
    for conf_thresh, distance_threshold, min_cluster_size in param_combinations:
        # The cluster function requires a unique index for cages
        train_preds_copy = train_preds.copy()
        train_preds_copy.reset_index(drop=True, inplace=True)
        train_preds_copy['index'] = train_preds_copy.index

        # Note that in this script we do just cluster by year because we care about measuring performance,
        # not the period-level tonnage estimates
        train_preds_copy.to_crs('EPSG:3035', inplace=True)

        train_facility_preds = predictions_cluster(
            train_preds_copy,
            facilities_path='',
            conf_thresh=conf_thresh,
            distance_threshold=distance_threshold,
            amnt_min_clusters=int(min_cluster_size),
            include_area=True,
            save=False,
            return_detections=True,
            cluster_variable='year'
        )

        # Evaluate performance at hyperparam.
        train_preds_copy.to_crs('EPSG:3857', inplace=True)
        stats = get_stats_total(
            labels=train_labels, preds=train_facility_preds.loc[train_facility_preds['det_conf'] >= conf_thresh])
        stats['conf_thresh'] = conf_thresh
        stats['distance_threshold'] = distance_threshold
        stats['min_cluster_size'] = min_cluster_size
        train_stats_dicts.append(stats)

    # Compute decision metrics.
    results = pd.DataFrame.from_records(train_stats_dicts)
    results['product'] = results['precision'] * results['recall']
    results['f_score'] = 2 * (results['product'] / (results['precision'] + results['recall']))

    # Get best hypeparameters based on decision metrics.
    product_best_id = results['product'].idxmax()
    product_train_best = results[['conf_thresh', 'distance_threshold', 'min_cluster_size']].loc[product_best_id].to_dict()

    f_score_best_id = results['f_score'].idxmax()
    f_score_train_best = results[['conf_thresh', 'distance_threshold', 'min_cluster_size']].loc[f_score_best_id].to_dict()

    # Test best hyperparameters.
    # * The cluster function requires a unique index for cages
    test_preds.reset_index(drop=True, inplace=True)
    test_preds['index'] = test_preds.index
    test_preds.to_crs('EPSG:3035', inplace=True)

    product_test_facility_preds = predictions_cluster(
        test_preds,
        facilities_path='',
        conf_thresh=product_train_best['conf_thresh'],
        distance_threshold=product_train_best['distance_threshold'],
        amnt_min_clusters=int(product_train_best['min_cluster_size']),
        include_area=False,
        save=False,
        return_detections=True,
        cluster_variable='year'
    )
    test_preds.to_crs('EPSG:3857', inplace=True)

    product_result = get_stats_total(
        labels=test_labels,
        preds=product_test_facility_preds.loc[product_test_facility_preds['det_conf'] >= product_train_best['conf_thresh']])

    test_preds.to_crs('EPSG:3035', inplace=True)
    f_score_test_facility_preds = predictions_cluster(
        test_preds,
        facilities_path='',
        conf_thresh=f_score_train_best['conf_thresh'],
        distance_threshold=f_score_train_best['distance_threshold'],
        amnt_min_clusters=int(f_score_train_best['min_cluster_size']),
        include_area=False,
        save=False,
        return_detections=True,
        cluster_variable='year'
    )
    test_preds.to_crs('EPSG:3857', inplace=True)
    f_score_result = get_stats_total(
        labels=test_labels,
        preds=f_score_test_facility_preds.loc[f_score_test_facility_preds['det_conf'] >= product_train_best['conf_thresh']])

    # Store results
    product_train_best = {'train_best_' + id: val for id, val in product_train_best.items()}
    product_result = {'test_' + id: val for id, val in product_result.items()}
    f_score_train_best = {'train_best_' + id: val for id, val in f_score_train_best.items()}
    f_score_result = {'test_' + id: val for id, val in f_score_result.items()}

    product_result.update(product_train_best)
    f_score_result.update(f_score_train_best)
    product_result['metric'] = 'product'
    f_score_result['metric'] = 'f_score'

    return [product_result, f_score_result]


def test_set_performance(
        images: gpd.GeoDataFrame,
        predictions: gpd.GeoDataFrame,
        labels: gpd.GeoDataFrame,
        confidence_threshold: float,
        distance_threshold: float,
        minimum_cluster_size: int):
    # Get the detections and labels
    test_preds = predictions[predictions['image'].isin(images['image'])].copy()
    test_labels = labels[labels['image'].isin(images['image'])].copy()

    # Post-process predictions using the hyperparameters
    test_preds.to_crs('EPSG:3035', inplace=True)
    test_facility_preds = predictions_cluster(
        test_preds, facilities_path='', conf_thresh=confidence_threshold, distance_threshold=distance_threshold,
        amnt_min_clusters=int(minimum_cluster_size), include_area=False, save=False, return_detections=True,
        cluster_variable='year'
    )
    test_facility_pred_df = predictions_cluster(
        test_preds, facilities_path='', conf_thresh=confidence_threshold,
        distance_threshold=distance_threshold, amnt_min_clusters=int(minimum_cluster_size),
        include_area=False, save=False, return_detections=False, cluster_variable='year'
    )
    test_preds.to_crs('EPSG:3857', inplace=True)

    # Cage-level performance (since we 1) labeled all the model detections and 2) since we found no labels in
    # stratum #7, we simply take the average fp to compute precision and recall across all detections and labels,
    # respectively.
    cage_result = get_stats_total(labels=test_labels, preds=test_facility_preds)

    # Facility-level performance
    test_labels.to_crs('EPSG:3035', inplace=True)
    test_labels['det_conf'] = 1.
    test_labels.reset_index(inplace=True, drop=True)
    test_labels['index'] = test_labels.index
    test_facility_label_df = predictions_cluster(
        test_labels, facilities_path='', conf_thresh=0,
        distance_threshold=distance_threshold,
        amnt_min_clusters=minimum_cluster_size,
        include_area=False, save=False, return_detections=False, cluster_variable='year'
    )
    test_labels.to_crs('EPSG:3857', inplace=True)

    # * Define facility geometries as the bounds of the cage clusters
    test_facility_label_df['all_cages'] = test_facility_label_df.apply(
        lambda row: row['square_farm_geoms'].buffer(0).union(row['circle_farm_geoms'].buffer(0)), axis=1)
    test_facility_label_df['bounds'] = test_facility_label_df['all_cages'].apply(lambda all_cages: box(*all_cages.bounds))
    test_facility_label_df.set_geometry('bounds', inplace=True, crs=test_facility_label_df.crs)

    test_facility_pred_df['all_cages'] = test_facility_pred_df.apply(
        lambda row: row['square_farm_geoms'].buffer(0).union(row['circle_farm_geoms'].buffer(0)), axis=1)
    test_facility_pred_df['bounds'] = test_facility_pred_df['all_cages'].apply(lambda all_cages: box(*all_cages.bounds))
    test_facility_pred_df.set_geometry('bounds', inplace=True, crs=test_facility_pred_df.crs)

    test_facility_pred_df['type'] = 'facility'
    test_facility_label_df['type'] = 'facility'
    facility_result = get_stats_total(labels=test_facility_label_df, preds=test_facility_pred_df)

    cage_result['type'] = 'cage-level'
    facility_result['type'] = 'facility-level'
    test_results = pd.concat([
        pd.DataFrame.from_dict(cage_result, orient='index'),
        pd.DataFrame.from_dict(facility_result, orient='index')], axis=1)
    return test_results


if __name__ == "__main__":
    # Get root directory
    root_dir = file_utils.get_root_path()

    # Load datasets for evaluation
    datasets = load_datasets_for_model_evaluation(
        detections_path=cfg.detections_path, landshape_path=cfg.landshape_path, base_dir=str(root_dir))
    all_images = datasets['all_images']
    ocean_images = datasets['ocean_images']
    ocean_detections = datasets['ocean_detections']
    labels = datasets['labels']

    # Create a separate test set
    ocean_images_train, ocean_images_test, _, _ = train_test_split(
        ocean_images, ocean_images['bucket'].cat.codes, test_size=0.1, random_state=cfg.random_state)

    if not os.path.exists(cfg.output_path):
        print('Running k-fold CV')
        # Create folds
        print('Create folds.')
        kfold = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.random_state)
        fold_indices = [fold for fold in kfold.split(ocean_images_train, ocean_images_train['bucket'].cat.codes)]

        # Run kfold evaluation
        with Pool(cfg.workers) as p:
            kfold_results = [
                [dict(result, fold_id=fold_id) for result in results]
                for fold_id, results in tqdm(
                    enumerate(
                        p.imap_unordered(
                            partial(
                                get_fold_performance,
                                images=ocean_images_train,
                                predictions=ocean_detections,
                                labels=labels
                            ),
                            fold_indices,
                            chunksize=cfg.chunksize,
                        )
                    ),
                    total=len(fold_indices),
                    desc=f"Running {cfg.num_folds}-fold evaluation"
                )
            ]
            p.close()
            p.join()

        # Save results
        kfold_results = itertools.chain.from_iterable(kfold_results)
        results = pd.DataFrame.from_records(kfold_results)
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        results.to_csv(cfg.output_path)

    # Test set performance using best HPs
    
    # Optimal hyperparameters found using above grid search
    CONF_THRESH = 0.785
    DISTANCE_THRESHOLD = 50
    MIN_CLUSTER_SIZE = 5
    
    print('Evaluating test set performance')
    test_df = test_set_performance(
        images=ocean_images_test, predictions=ocean_detections, labels=labels,
        confidence_threshold=CONF_THRESH, distance_threshold=DISTANCE_THRESHOLD, minimum_cluster_size=MIN_CLUSTER_SIZE)
    test_df.to_csv(cfg.output_path.replace('fold_results', 'test_results'))
