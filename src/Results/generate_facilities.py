"""
Generates the AquaFacility object instances (see class in utils_tonnage) for the predictions and
for the CloudFactory annotations.
"""

import argparse
import os

from src.utils import load_cf_labels, map_year_to_image_pass_opt2, load_final_image_boxes, CRS_DICT
from src.cluster_facilities import DBSCAN_cluster

# Tonnage estimates functions
from src.utils_tonnage import (
    AquaFacility, load_AquaFacility, CF_Facility,
    compute_cage_area_estimates_gdf, dedup_cages_in_overlap_years_with_white_space
)
import src.file_utils as file_utils


def generate_Prediction_Facility(
        facility_file_name: str,
        conf_thresh: float,
        distance_threshold: float,
        min_cluster_size: int,
        min_cage_threshold: float,
        default_cage_depth: float,
        bathymetry_statistic: str
) -> AquaFacility:
    root_dir = file_utils.get_root_path()
    if not os.path.exists(f'{root_dir}/output/Facilities/{facility_file_name}.pkl'):
        os.makedirs(f'{root_dir}/output/Facilities', exist_ok=True)

        print('[INFO] Generating facility from predictions..')

        Facility = load_AquaFacility(
            filename=None,
            main_dir=root_dir, selected_map=map_year_to_image_pass_opt2, image_selection='random',
            confidence_threshold=conf_thresh, distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size, time_group='pass'
        )

        Facility.compute_min_max_cages()
        Facility.add_depth(
            min_cage_threshold=min_cage_threshold, default_cage_depth=default_cage_depth,
            bathymetry_statistic=bathymetry_statistic)

        Facility.save(file=f'{root_dir}/output/Facilities/{facility_file_name}.pkl')

    print('[INFO] Loading facility...')
    Facility = load_AquaFacility(
        filename=f'{root_dir}/output/Facilities/{facility_file_name}.pkl',
        main_dir=None, selected_map=None, image_selection=None, confidence_threshold=None,
        distance_threshold=None, min_cluster_size=None, time_group=None
    )
    return Facility


def generate_CF_Facility(
        facility_file_name: str,
        distance_threshold: float,
        min_cluster_size: int,
        min_cage_threshold: float,
        default_cage_depth: float,
        bathymetry_statistic: str
) -> AquaFacility:
    root_dir = file_utils.get_root_path()
    if not os.path.exists(f'{root_dir}/output/Facilities/{facility_file_name}.pkl'):
        os.makedirs(f'{root_dir}/output/Facilities', exist_ok=True)

        print('[INFO] Generating facility from CF annotations..')
        # Load CF Labels
        cf_labels = load_cf_labels().rename(columns={"fn": "image"})

        # Load image boxes
        rmblank_image_boxes = load_final_image_boxes(main_dir=str(root_dir))

        # Add pass and unique identifier for the cages
        cf_preds = cf_labels.copy()

        # Drop flag annotations
        cf_preds = cf_preds.loc[cf_preds['type'] != 'flag']

        # Rename farm types to match names required for facility clustering
        cf_preds['type'] = cf_preds['type'].map({'circle_cage': 'circle_farm', 'square_cage': 'square_farm'})
        cf_preds['farm_type'] = cf_preds['type']
        cf_preds['pass'] = cf_preds['year'].map(map_year_to_image_pass_opt2)
        cf_preds.reset_index(drop=True, inplace=True)
        cf_preds['index'] = cf_preds.index

        # Add cage area, min area, max area
        cf_preds = compute_cage_area_estimates_gdf(gdf=cf_preds, bounds=True)

        # Deduplicate predictions
        cf_preds.drop('bbox_ind', axis=1, inplace=True)
        cages, annual_coverage = dedup_cages_in_overlap_years_with_white_space(
            cages=cf_preds, image_boxes=rmblank_image_boxes, pass_map=map_year_to_image_pass_opt2,
            year_selection='random')

        # Cluster cages into facilities
        cf_preds.to_crs(f'EPSG:{CRS_DICT["area"]}', inplace=True)
        cf_final_facilities = DBSCAN_cluster(
            cf_preds,
            facilities_path='',
            distance_threshold=distance_threshold,
            amnt_min_clusters=min_cluster_size,
            include_area=True,
            save=False,
            return_detections=False,
            cluster_variable='pass'
        )
        cf_preds.to_crs(f'EPSG:{CRS_DICT["mapping"]}', inplace=True)

        # Set up Facility object
        Facility = CF_Facility(
            final_facilities=cf_final_facilities, preds=cf_preds, cages=cages, annual_coverage=annual_coverage,
            selected_map=map_year_to_image_pass_opt2, distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size, rmblank_image_boxes=rmblank_image_boxes, image_selection='random',
            main_dir=str(root_dir)
        )

        Facility.compute_min_max_cages()
        Facility.add_depth(
            min_cage_threshold=min_cage_threshold, default_cage_depth=default_cage_depth,
            bathymetry_statistic=bathymetry_statistic)

        Facility.save(file=f'{root_dir}/output/Facilities/{facility_file_name}.pkl')

    print('[INFO] Loading facility...')
    Facility = load_AquaFacility(
        filename=f'{root_dir}/output/Facilities/{facility_file_name}.pkl', main_dir=None, selected_map=None,
        image_selection=None, confidence_threshold=None,
        distance_threshold=None, min_cluster_size=None, time_group=None
    )
    return Facility


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_thresh', type=float)
    parser.add_argument('--min_cage_threshold', type=float)
    parser.add_argument('--distance_threshold', type=float)
    parser.add_argument('--min_cluster_size', type=int)
    parser.add_argument('--default_cage_depth', type=float)
    parser.add_argument('--bathymetry_statistic', type=str)
    args = parser.parse_args()

    PredictionFacility = generate_Prediction_Facility(
        facility_file_name='AQ_tunedfacility',
        conf_thresh=args.conf_thresh,
        distance_threshold=args.distance_threshold,
        min_cluster_size=args.min_cluster_size,
        min_cage_threshold=args.min_cage_threshold,
        default_cage_depth=args.default_cage_depth,
        bathymetry_statistic=args.bathymetry_statistic
    )

    LabelFacility = generate_CF_Facility(
        facility_file_name='CF_Facility',
        distance_threshold=args.distance_threshold,
        min_cluster_size=args.min_cluster_size,
        min_cage_threshold=args.min_cage_threshold,
        default_cage_depth=args.default_cage_depth,
        bathymetry_statistic=args.bathymetry_statistic
    )
