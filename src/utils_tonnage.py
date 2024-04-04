from typing import Dict, List, Callable
from itertools import permutations as iter_permutations
import matplotlib.axes
import numpy as np
import geopandas as gpd
import os
import pandas as pd
import rasterio
import shapely
from rasterstats import zonal_stats
from scipy.stats import truncnorm, norm
from shapely.ops import unary_union
from shapely.geometry import polygon
from tqdm import tqdm
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pickle

from src.process_yolo.calc_net_areas import get_circle_area_from_bbox, get_square_area_from_bbox
from src.utils import (
    deduplicate_gdf_with_bboxes, generate_image_specs_from_file_name, load_final_image_boxes, load_download_bboxes,
    deduplicate_download_boxes, load_cf_labels, get_cage_min_and_max_areas, CRS_DICT
)
from src.cluster_facilities import DBSCAN_cluster


def compute_facility_tonnage_estimates(
        facility_df: gpd.GeoDataFrame,
        period_factor_table: pd.DataFrame,
        period_var: str,
        min_cage_threshold: float,
        preds_df: pd.DataFrame,
        model_error_distributions: pd.DataFrame,
        depth_dist_mixture_param: float,
        K: int = 0) -> pd.DataFrame:
    """
    Computes the tonnage and tonnage variance at the facility level.
    :param preds_df: GDF of cages that corresponds to the unique ID of each cage (to sample model errors)
    :param min_cage_threshold:
    :param facility_df: gdf of facilities including area, area variance, minimum area, maximum area
    :param period_factor_table: period-level expectation and sd for depth, stocking density and harvest
    frequency parameters
    :param period_var: ['year', 'pass']
    :param K number of simulations for bootstrap method
    :param model_error_distributions: pd.DataFrame indicating the model error distributions (mean, sd)
    for each (pass, cage_type) combination. Computed by the define_model_error_distributions function.
    Assumes that the mean and standard deviation are those of a Normal distribution.
    :return: gdf including facility-level tonnage and tonnage variance
    """
    # Add h, s, d parameters to each facility (according to the year)
    estimation_df = facility_df.merge(
        period_factor_table, how='left', on=[period_var], validate='many_to_one')

    # Set up cages min and cages max
    tonnage = {period: [] for period in estimation_df['pass'].unique()}
    for k in tqdm(range(K)):
        # Compute
        sim_df = estimation_df.copy()

        # Sample model prediction error: this function replaces the area, min_area and max_area columns
        # by sampling from the model error distributions. This also incorporates uncertainty
        # in the image selection, so that ultimately expand the min_area and max_area bounds by using
        # the min_area and max_area for the minimum area image selection and maximum area image selection,
        # respectively.
        sim_df = sample_model_errors(
            facility_df=sim_df, cages_df=preds_df,
            model_error_distributions=model_error_distributions)

        # Sample Uniform areas
        sim_df['sim_area'] = np.random.uniform(
            low=sim_df['min_area'], high=sim_df['max_area'])

        # Sample depth: We combine two truncated Normals. With probability depth_dist_mixture_param, we
        # use the TN that samples from the region [min_cage_threshold, bathy/2], with probability
        # 1-depth_dist_mixture_param, we use the TN that samples from [bathy/2, bathy].
        # Note that we don't use the bathymetry variable directly, but rather cage_depth * 2, due to the thresholding
        # that we impose on the bathymetry data.
        sim_df['depth_Bernoulli'] = np.random.binomial(n=1, p=depth_dist_mixture_param, size=len(sim_df))
        sim_df['sim_depth_dA'] = truncnorm.rvs(
            loc=sim_df['cage_depth'],  # Center at bathymetry depth / 2
            scale=(sim_df['cage_depth'] - min_cage_threshold) / 1.96,
            a=(min_cage_threshold - sim_df['cage_depth']) / ((sim_df['cage_depth'] - min_cage_threshold) / 1.96),
            b=0  # Upper bound is bathymetry depth / 2
        )
        sim_df['sim_depth_dB'] = truncnorm.rvs(
            loc=sim_df['cage_depth'],  # Center at bathymetry depth / 2
            scale=sim_df['cage_depth'] / 1.96,
            a=0,
            b=sim_df['cage_depth'] / (sim_df['cage_depth'] / 1.96)
        )
        sim_df['sim_depth'] = np.where(
            sim_df['depth_Bernoulli'] == 1, sim_df['sim_depth_dA'], sim_df['sim_depth_dB'])

        # Sample stocking density (truncated normal)
        stockingd_bounds = (5, 20)
        sim_df['sim_stockingd'] = truncnorm.rvs(
            loc=sim_df['s_mean'], scale=sim_df['s_sd'],
            a=(stockingd_bounds[0] - sim_df['s_mean']) / sim_df['s_sd'],
            b=(stockingd_bounds[1] - sim_df['s_mean']) / sim_df['s_sd']
        )

        # Sample harvest frequency
        sim_df['sim_harvestf'] = np.random.normal(
            loc=sim_df['h_mean'], scale=sim_df['h_sd'])

        # Compute tonnage
        sim_df['sim_tonnage'] = sim_df['sim_area'] * sim_df['sim_depth'] * sim_df['sim_stockingd']
        sim_df['sim_tonnage'] *= sim_df['sim_harvestf'] * (1 / 1000)

        sim_periods = sim_df.groupby('pass')['sim_tonnage'].sum()
        for period in tonnage.keys():
            tonnage[period].append(sim_periods[period])

    # Compute point estimate and variance for each period
    period_estimates = pd.DataFrame()
    for period in tonnage.keys():
        period_estimates = pd.concat([
            period_estimates,
            pd.DataFrame.from_dict(
                {'pass': [period], 'tonnage': [np.mean(tonnage[period])], 'tonnage_var': np.var(tonnage[period])})
        ])
    period_estimates['tonnage_sd'] = period_estimates['tonnage_var'].apply(lambda var: np.sqrt(var))
    period_estimates.set_index('pass', inplace=True)
    period_estimates = period_estimates.sort_index()

    return period_estimates


def define_model_error_distributions(
        cf_labels: gpd.GeoDataFrame,
        detections_df: gpd.GeoDataFrame,
        dedup_boxes: gpd.GeoDataFrame,
        selected_map: Callable,
        confidence_threshold: float,
        visualize: str = None

) -> pd.DataFrame:
    """
    Dataframe including (model_error_mean) and (model_error_sd), the parameters of a Normal distribution for
    the model errors for each cage type and period combination.
    :param visualize: if not None, saves visualizations of the model errors to this path
    :param confidence_threshold: model confidence threshold to filter detections
    :param selected_map: function mapping years to periods
    :param dedup_boxes: deduplicated boxes
    :param cf_labels: raw df of Cloud Factory labels
    :param detections_df: raw df of ocean cage detections
    :return: pd.DataFrame indicating the model error distributions (mean, sd) for each (pass, cage_type) combination
    """

    # Filter labels and get bbox ind
    cf_labels = cf_labels.loc[cf_labels['type'].isin(['circle_cage', 'square_cage'])].copy()
    cf_labels['bbox_ind'] = cf_labels['image'].apply(lambda f: f.split('_')[1])
    detections_df['bbox_ind'] = detections_df['image'].apply(lambda f: f.split('_')[1])

    # De-duplicate only spatially so as to not double count errors for some cages
    # Note that pred_df is already de-duplicated in geocode_results.py so this isn't really needed
    print('[INFO] De-duplicating labels and predictions')
    cf_labels = deduplicate_gdf_with_bboxes(dedup_boxes, cf_labels, path=None)
    detections_df = deduplicate_gdf_with_bboxes(dedup_boxes, detections_df, path=None)

    # Add period
    cf_labels['pass'] = cf_labels['year'].astype(int).map(selected_map)
    detections_df['pass'] = detections_df['year'].astype(int).map(selected_map)

    # Compute cage area estimates for CF labels
    cf_labels = compute_cage_area_estimates_gdf(gdf=cf_labels)

    # Filter detections by confidence threshold
    detections_df = detections_df[detections_df['det_conf'] > confidence_threshold]

    # Compute error distribution for each period - cage type combination
    if visualize:
        fig, axs = plt.subplots(len(cf_labels['pass'].unique()), 2, figsize=(15, 15), sharex=True)

    model_distribution_errors = pd.DataFrame()
    period_passes = list(cf_labels['pass'].unique())
    period_passes.sort()
    for i, period_pass in enumerate(period_passes):
        for j, cage_type in enumerate(['circle', 'square']):
            labels = cf_labels.loc[(cf_labels['pass'] == period_pass) & (cf_labels['type'] == f'{cage_type}_cage')]
            sample_preds = detections_df.loc[
                (detections_df['pass'] == period_pass) & (detections_df['type'] == f'{cage_type}_farm')]

            query = sample_preds[['year', 'geometry', 'area']].copy()
            key = labels[['year', 'geometry', 'area']].copy()

            # Get error distribution
            errors = get_cage_area_errors_from_labels(query=query, key=key)
            mu, std = norm.fit(errors)
            model_distribution_errors = pd.concat([
                model_distribution_errors,
                pd.DataFrame.from_dict({
                    'pass': [period_pass], 'farm_type': [f'{cage_type}_farm'],
                    'model_error_mean': [mu], 'model_error_sd': [std]})
            ])
            if visualize:
                plot_errors_and_normal(errors, ax=axs[i, j])
                axs[i, j].title.set_text(f'{period_pass} - {cage_type} farm')
    if visualize:
        plt.savefig(visualize)

    return model_distribution_errors


def compute_cage_area_estimates_gdf(gdf: gpd.GeoDataFrame, bounds=False) -> gpd.GeoDataFrame:
    """
    Computes area estimates for the square and circle cages in a geodataframe. The gdf
    must include the original image's height and width ['jpeg_height', 'jpeg_width'],
    the 'type' of cage ['circle_cage', 'square_cage'] and the location of
    the annotation ['xmin', 'xmax', 'ymin', 'ymax']
    :param gdf: gdf of cages
    :param bounds: If true, returns the min and max cage area as well
    :return: gdf including 'area' column with estimates for circle and square cages
    """

    gdf_areas = gdf.copy()
    # Convert to EPSG:3035 to compute the area of the cages
    gdf_areas = gdf_areas.to_crs(f'EPSG:{CRS_DICT["area"]}')

    areas = []
    area_vars = []
    min_areas = []
    max_areas = []
    for _, row in tqdm(gdf_areas.iterrows(), total=len(gdf_areas)):
        height, width = row['jpeg_height'], row['jpeg_width']
        xmin_m, ymin_m, xmax_m, ymax_m = row['geometry'].bounds

        bbox_width_m = xmax_m - xmin_m
        bbox_height_m = ymax_m - ymin_m

        area, area_var, min_area, max_area = 0, 0, 0, 0
        if row['type'] in ['circle_cage', 'circle_farm']:
            # check if its on the border
            x_border = False
            y_border = False
            if row['xmin'] == 0 or row['xmax'] == width:
                x_border = True

            if row['ymin'] == 0 or row['ymax'] == height:
                y_border = True

            area, area_var, min_area, max_area = get_circle_area_from_bbox(
                bbox_width_m, bbox_height_m, x_border=x_border, y_border=y_border)

        elif row['type'] in ['square_cage', 'square_farm']:
            area, area_var, min_area, max_area = get_square_area_from_bbox(bbox_width_m, bbox_height_m)

        areas.append(area)
        area_vars.append(area_var)
        min_areas.append(min_area)
        max_areas.append(max_area)

    gdf_areas['area'] = areas
    if bounds:
        gdf_areas['area_var'] = area_vars
        gdf_areas['min_area'] = min_areas
        gdf_areas['max_area'] = max_areas
    # Convert back to EPSG:3857
    gdf_areas = gdf_areas.to_crs(f'EPSG:{CRS_DICT["mapping"]}')
    return gdf_areas


def plot_errors_and_normal(errors: pd.Series, ax: matplotlib.axes.Axes) -> None:
    """
    Plots a histogram of errors and the Normal distribution fit to these using scipy.stats.norm
    :param errors:
    :param ax:
    """
    # Fit normal
    mu, std = norm.fit(errors)

    # Plot the histogram.
    ax.hist(errors, bins=10, density=True, alpha=0.6, color='g')

    # Plot the PDF
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.axvline(x=0)


def get_cage_area_errors_from_labels(query: gpd.GeoDataFrame, key: gpd.GeoDataFrame) -> pd.Series:
    """
    Computes the distribution of errors, or differences in cage area between the query and the
    key. It finds the polygon from the key with the highest degree of overlap to the query, and
    reports the difference between its area and that of the object in the query.
    :param query: a gdf with columns ['area', 'geometry', 'year']
    :param key: a gdf with columns ['area', 'geometry', 'year']
    :return: errors (key object area - query object area)
    """
    # Query and key should have year, geometry and area columns
    query.reset_index(drop=True, inplace=True)
    query['index'] = query.index
    key['geometry_key'] = key['geometry']

    # Since we're computing areas, convert to EPSG:3035
    query_crs, key_crs = query.crs, key.crs
    query.to_crs(f"EPSG:{CRS_DICT['area']}", inplace=True)
    key.to_crs(f"EPSG:{CRS_DICT['area']}", inplace=True)

    # NOTE: for now, we're keeping errors only for cages that did match
    # Spatial join (only cages for the same year)
    joined = query.sjoin(
        key, how='left', predicate='intersects', lsuffix='query', rsuffix='key')
    joined = joined.loc[~joined['year_key'].isna()]
    joined = joined.loc[joined['year_query'] == joined['year_key']]

    # Out of the cages with multiple matches, keep the one from the match with the highest spatial overlap
    joined['spatial_overlap'] = joined.apply(
        lambda row: None if pd.isnull(row['geometry_key']) else row['geometry'].intersection(row['geometry_key']).area /
                                                                row['geometry'].area * 100,
        axis=1)
    joined = joined.sort_values('spatial_overlap', ascending=False)
    joined = joined.drop_duplicates(subset=['index'], keep='first')

    # Compute area difference
    joined['area_key'] = joined['area_key'].fillna(value=0)
    joined['area_dif'] = joined['area_key'] - joined['area_query']

    errors = joined['area_dif']

    # Convert back
    query.to_crs(query_crs, inplace=True)
    key.to_crs(key_crs, inplace=True)
    return errors


def sample_model_errors(
        facility_df: pd.DataFrame,
        cages_df: pd.DataFrame,
        model_error_distributions: pd.DataFrame) -> pd.DataFrame:
    """
    Samples from the model error distributions for each pass and farm type to compute area, min_area and max_area
    values for each facility that incorporate this uncertainty.
    :param model_error_distributions: pd.DataFrame indicating the model error distributions (mean, sd)
    for each (pass, cage_type) combination. Computed by the define_model_error_distributions function.
    Assumes that the mean and standard deviation are those of a Normal distribution.
    :param facility_df: Dataframe of facilities
    :param cages_df: Dataframe of predicted cages
    :return: A facility dataframe with new area, min_area and max_area values (with the previous values
    in the _og columns) that incorporate model performance uncertainty.
    """

    assert 'facility_index' in facility_df.columns and 'cage_ids' in facility_df.columns
    assert 'cage_ids_min' in facility_df.columns and 'cage_ids_max' in facility_df.columns

    multi_selection_df = pd.DataFrame()
    for year_selection in ['min', 'max', 'random']:
        sim_selection = facility_df.copy()
        if year_selection != 'random':
            sim_selection.drop('cage_ids', inplace=True, axis=1)
            sim_selection.rename(columns={f'cage_ids_{year_selection}': 'cage_ids'}, inplace=True)
        sim_selection['year_selection'] = year_selection
        multi_selection_df = pd.concat([multi_selection_df, sim_selection[['facility_index', 'cage_ids', 'year_selection']]])

    # Pivot to have indicators of which cages exist under each selection type. We need to explode here
    # because we can't pivot using a cage_ids list as an index.
    multi_selection_df['value'] = 1
    multi_selection_df = multi_selection_df.explode('cage_ids')
    multi_selection_df = multi_selection_df.loc[~multi_selection_df['cage_ids'].isna()]

    multi_selection_df['cage_ids'] = multi_selection_df['cage_ids'].astype(int)
    multi_selection_df = multi_selection_df.pivot(
        index=['facility_index', 'cage_ids'], columns='year_selection', values='value').reset_index()

    multi_selection_df[['min', 'max', 'random']] = multi_selection_df[['min', 'max', 'random']].fillna(0)
    multi_selection_df = multi_selection_df.astype({'min': 'int', 'max': 'int', 'random': 'int'})
    assert multi_selection_df[['max', 'min', 'random']].max().max() == 1

    # At this stage, we have everything at the facid-cageid level so that each unique cage gets the same
    # perturbation across year_selection.

    # Merge facility and cage information
    fac_cages = multi_selection_df.merge(
        cages_df[['index', 'farm_type', 'pass', 'area', 'area_var', 'min_area', 'max_area']],
        how='left', left_on='cage_ids', right_on='index', validate='one_to_one')
    fac_cages.rename(
        columns={'area': 'area_orig', 'area_var': 'area_var_orig', 'min_area': 'min_area_og',
                 'max_area': 'max_area_og'}, inplace=True)

    # Sample errors and get new area
    fac_cages = fac_cages.merge(model_error_distributions, how='left', on=['pass', 'farm_type'], validate='many_to_one')
    fac_cages['model_error'] = np.random.normal(
        loc=fac_cages['model_error_mean'], scale=fac_cages['model_error_sd'])
    fac_cages['area'] = fac_cages['area_orig'] + fac_cages['model_error']

    # Ensure that cage area is positive: we keep resampling errors for cage areas that are < 0
    while fac_cages['area'].min() <= 0:
        fac_cages['model_error'] = np.random.normal(
            loc=fac_cages['model_error_mean'], scale=fac_cages['model_error_sd'])
        fac_cages['area'] = np.where(
            fac_cages['area'] <= 0, fac_cages['area_orig'] + fac_cages['model_error'], fac_cages['area'])

    # Redefine area min and area max
    fac_cages['min_area'] = None
    fac_cages['max_area'] = None
    # * Complete ellipses do not have variance
    fac_cages['min_area'] = np.where(
        (fac_cages['farm_type'] == 'circle_farm') & (fac_cages['area_var_orig'] == 0.),
        fac_cages['area'], fac_cages['min_area'])
    fac_cages['max_area'] = np.where(
        (fac_cages['farm_type'] == 'circle_farm') & (fac_cages['area_var_orig'] == 0.),
        fac_cages['area'], fac_cages['max_area'])
    # * Incomplete ellipses
    fac_cages['min_area'] = np.where(
        (fac_cages['farm_type'] == 'circle_farm') & (fac_cages['area_var_orig'] != 0.),
        4 * fac_cages['area'] / (2 + np.pi), fac_cages['min_area'])
    fac_cages['max_area'] = np.where(
        (fac_cages['farm_type'] == 'circle_farm') & (fac_cages['area_var_orig'] != 0.),
        2 * np.pi * fac_cages['area'] / (2 + np.pi), fac_cages['max_area'])
    # * Square cages
    fac_cages['min_area'] = np.where(
        (fac_cages['farm_type'] == 'square_farm'),
        2 * fac_cages['area'] / 3, fac_cages['min_area'])
    fac_cages['max_area'] = np.where(
        (fac_cages['farm_type'] == 'square_farm'),
        4 * fac_cages['area'] / 3, fac_cages['max_area'])
    fac_cages['min_area'] = fac_cages['min_area'].astype(float)
    fac_cages['max_area'] = fac_cages['max_area'].astype(float)

    # Aggregate back to the facility level: first we sum across cages
    fac_cages = fac_cages.\
        groupby(['facility_index', 'min', 'max', 'random'])[['area', 'min_area', 'max_area']].\
        agg('sum').reset_index()

    # Back to long format so we can aggregate all the cage groups of the facility
    fac_cages = pd.melt(fac_cages, id_vars=['facility_index', 'area', 'min_area', 'max_area'],
                        value_vars=['min', 'max', 'random'], var_name='year_selection', value_name='value')
    fac_cages = fac_cages.loc[fac_cages['value'] == 1]

    # Now we obtain the maximum and min areas for each facility: from the min year selection, we use the min
    # area, for the max year selection we use the max area to widen the bounds of the Uniform. We keep the area
    # from random just as a point estimate but this is not used in the bootstrap so is actually unnecessary.
    fac_cages = fac_cages.groupby(
        ['facility_index', 'year_selection'])[['area', 'min_area', 'max_area']].sum().reset_index()
    fac_cages['selected_area'] = None
    fac_cages['selected_area'] = np.where(
        fac_cages['year_selection'] == 'min', fac_cages['min_area'], fac_cages['selected_area'])
    fac_cages['selected_area'] = np.where(
        fac_cages['year_selection'] == 'max', fac_cages['max_area'], fac_cages['selected_area'])
    fac_cages['selected_area'] = np.where(
        fac_cages['year_selection'] == 'random', fac_cages['area'], fac_cages['selected_area'])

    # Back to wide format so we can just append the columns to our main data frame
    fac_cages = fac_cages.pivot(index='facility_index', columns='year_selection', values='selected_area').reset_index()
    fac_cages.rename(columns={'max': 'max_area', 'min': 'min_area', 'random': 'area'}, inplace=True)

    # For facilities with zero cages under the min year selection, set a min area of zero
    fac_cages['min_area'] = fac_cages['min_area'].fillna(0)

    # Add new area, min area and max area to facility_df
    facility_df.rename(
        columns={'area': 'area_orig', 'area_var': 'area_var_orig', 'min_area': 'min_area_og',
                 'max_area': 'max_area_og'}, inplace=True)
    fac_cages = facility_df.merge(fac_cages, how='left', on='facility_index', validate='one_to_one')
    return fac_cages


def load_production_factors(prod_file: str) -> pd.DataFrame:
    """
    Generates a table with the factor estimates and standard deviations of each type of parameter, across
    species
    :param prod_file: path to Excel file with production parameters
    :return: pd.DataFrame including mean and standard deviation for each parameter
    """
    # Final columns
    cols = ['Species', 'Parameter', 'Factor', 'Range (lower)', 'Range (upper)']

    # Standard deviations for the annual harvest frequency of each species.
    production_factors = pd.read_excel(
        prod_file, sheet_name="Production Factors")
    production_factors["Species"] = production_factors["Species"].str.lower()

    # Stocking densities
    stocking_density_factors = production_factors.loc[
        production_factors['Factor Type'] == 'Stocking density'].copy()
    stocking_density_factors['Parameter'] = 'Stocking density'
    stocking_density_factors = stocking_density_factors[cols]

    # Harvest frequencies
    harvest_frequency_factors = production_factors.loc[
        production_factors['Factor Type'] == 'Harvest frequency'].copy()
    # * Convert from months to annual frequency
    harvest_frequency_factors['Factor'] = harvest_frequency_factors['Factor'].apply(lambda f: 12 / f)
    harvest_frequency_factors['Range (lower)'] = harvest_frequency_factors['Range (lower)'].apply(lambda f: 12 / f)
    harvest_frequency_factors['Range (upper)'] = harvest_frequency_factors['Range (upper)'].apply(lambda f: 12 / f)
    harvest_frequency_factors['Units'] = 'Annual frequency'
    harvest_frequency_factors['Parameter'] = 'Annual harvest frequency'

    # Concatenate parameters
    stocking_density_factors = stocking_density_factors[cols]
    harvest_frequency_factors = harvest_frequency_factors[cols]
    factor_table = pd.concat([stocking_density_factors, harvest_frequency_factors])

    # Compute standard deviation (assuming a Uniform distribution over the range)
    factor_table['Standard deviation'] = factor_table.apply(
        lambda row: np.sqrt(np.power(row['Range (upper)'] - row['Range (lower)'], 2) / 12), axis=1)

    return factor_table


def load_fao_french_mediterranean(fao_file: str, pass_map: Callable) -> pd.DataFrame:
    """
    Generates a DataFrame including FAO data for French Mediterranean: production quantity (tonnage)
    for seabass, seabream and meagre during 2000-2020
    :rtype: pd.DataFrame
    """
    fao_data = pd.read_csv(fao_file, header=0)

    # Drop Totals and Information row
    fao_data = fao_data.loc[~fao_data['FAO major fishing area (Name)'].isna()]

    fao_data.rename({'ASFIS species (Name)': 'species'}, axis=1, inplace=True)
    fao_data.drop(fao_data.filter(regex='S').columns, axis=1, inplace=True)
    fao_data_long = fao_data.melt(
        id_vars=[
            'Country (Name)',
            'FAO major fishing area (Name)',
            'Environment (Name)',
            'species',
            'Unit (Name)', 'Unit'],
        var_name='year',
        value_name='production_quantity'
    )
    fao_data_long['year'] = fao_data_long['year'].apply(lambda y: y.replace('[', '').replace(']', ''))
    fao_data_long['year'] = fao_data_long['year'].astype(int)

    # Drop years before 2000 and map to periods
    fao_data_long = fao_data_long.loc[fao_data_long['year'] >= 2000]
    fao_data_long['pass'] = fao_data_long['year'].map(pass_map)

    # Clean species groups
    species_groups = ['seabass', 'seabream', 'meagre']
    fao_data_long["species_group"] = None

    for group in species_groups:
        fao_data_long.loc[fao_data_long['species'].str.contains(group, case=False), "species_group"] = group
    fao_data_long.loc[fao_data_long['species'].str.contains('marine fishes nei',
                                                            case=False), "species_group"] = 'seabream'
    fao_data_long = fao_data_long[fao_data_long['species_group'].isin(species_groups)]

    # Compute annual production shares
    fao_data_long['production_share'] = fao_data_long['production_quantity'] / fao_data_long.groupby(['year'])[
        'production_quantity'].transform(sum)

    # Fill missing values with zero for production quantities and production shares
    fao_data_long['production_quantity'] = fao_data_long['production_quantity'].fillna(0)
    fao_data_long['production_share'] = fao_data_long['production_share'].fillna(0)
    return fao_data_long


def generate_period_production_factors(
        production_factor_table: pd.DataFrame, fao_data: pd.DataFrame, period_var: str) -> pd.DataFrame:
    """
    Determines the factor and factor standard deviations for depth, stocking density and
    annual harvest frequency on an annual basis, using production share weights.
    :param production_factor_table: DataFrame including species-level parameters
    :param fao_data: production shares by species 2000-2020
    :param period_var: whether to group using year or period
    :return: pd.DataFrame
    """
    # Aggregate production quantity and shares by period
    fao_data_agg = fao_data.groupby([period_var, 'species_group'])[
        ['production_quantity']].sum().reset_index()
    fao_data_agg['production_share'] = fao_data_agg.apply(
        lambda row: row['production_quantity'] / fao_data_agg.groupby(period_var)['production_quantity'].sum()[
            row[period_var]], axis=1
    )

    period_factor_table = fao_data_agg.merge(
        production_factor_table, left_on='species_group', right_on='Species', how='left')

    # Compute parameters and parameter standard deviation (weighted by production shares)
    wm = lambda x: np.average(x, weights=period_factor_table.loc[x.index, "production_share"])
    wsd = lambda x: np.sqrt(np.dot(np.power(x, 2), np.power(period_factor_table.loc[x.index, "production_share"], 2)))

    period_factor_table = period_factor_table.groupby([period_var, 'Parameter']).agg(
        {'Factor': wm, 'Standard deviation': wsd}
    ).reset_index()

    # Dcast
    period_factor_table = period_factor_table.pivot_table(
        index=[period_var], columns=['Parameter'], values=['Factor', 'Standard deviation']).reset_index()
    period_factor_table.columns = [f'{x} {y}' if y != '' else x for x, y in period_factor_table.columns]

    return period_factor_table


def add_facility_depth(
        facility_df: gpd.GeoDataFrame,
        bathymetry_path: str,
        min_cage_threshold: float,
        default_cage_depth: float,
        bathymetry_statistic: str
) -> gpd.GeoDataFrame:
    """
    Determines cage depth at the level of each facility using bathymetry data.
    :param min_cage_threshold: Minimum depth level for the cages (meters)
    :param facility_df: gpd of aquaculture facilities
    :param bathymetry_path: location of the bathymetry tif from EMOD to be used
    to perform the spatial join to facilities
    :param bathymetry_statistic whether to use 'bathy_min' or 'bathy_depth' to represent a facility's
    water depth. 'bathy_min' uses the greatest depth, while 'bathy_depth' uses the mean. Note that
    bathymetry data is negative, which is why we use the min to maximize the depth.
    :return:
    """
    # Check bathymetry statistic
    assert bathymetry_statistic in ['bathy_depth', 'bathy_min']

    # Set an index so we can merge back information later
    facilities_depth = facility_df.copy()

    # Get the complete polygon of cages (including all cage types) for each facility
    facilities_depth['total_cage_geometry'] = facilities_depth.apply(
        lambda row: unary_union([row['circle_farm_geoms'], row['square_farm_geoms']]), axis=1)

    facilities_depth = facilities_depth[['total_cage_geometry']].copy()
    facilities_depth.rename(columns={'total_cage_geometry': 'geometry'}, inplace=True)
    facilities_depth = gpd.GeoDataFrame(facilities_depth, geometry='geometry', crs=facility_df.crs)

    # Compute bathymetry average for each facility
    facilities_depth.to_crs('EPSG:4326', inplace=True)
    with rasterio.open(bathymetry_path) as src:
        # Get no data value
        nodata = src.nodata

        fac_stats = pd.DataFrame(zonal_stats(
            facilities_depth,
            src.read(1),
            all_touched=True,
            affine=src.transform,
            nodata=nodata
        ))
    facilities_depth = pd.concat([facilities_depth, fac_stats], axis=1)

    # Clean outputs: if max == None, we use min. If still unavailable, we drop
    facilities_depth['bathy_depth'] = facilities_depth['mean']
    facilities_depth.rename(columns={'min': 'bathy_min', 'max': 'bathy_max', 'mean': 'bathy_mean'}, inplace=True)
    # Convert depth from negative to positive values
    for col in ['bathy_min', 'bathy_max', 'bathy_mean', 'bathy_depth']:
        facilities_depth[col] = facilities_depth[col] * -1

    # Compute cage depth. Per FAO guidelines, "the water depth should be 2 times the overall net depth"
    # See https://www.fao.org/3/i6719e/i6719e.pdf
    facilities_depth['cage_depth'] = facilities_depth[bathymetry_statistic].apply(
        lambda depth: default_cage_depth if pd.isnull(depth) else depth / 2)
    # Set a minimum cage depth threshold
    facilities_depth['cage_depth'] = facilities_depth['cage_depth'].apply(
        lambda depth: min_cage_threshold if depth <= min_cage_threshold else depth)
    print(f"[INFO] No bathymetry data available for {facilities_depth['bathy_depth'].isna().sum()} "
          f"facilities, using default depth")

    # Concatenate with input facility gdf
    facilities_depth = pd.concat(
        [facility_df,
         facilities_depth[[
             'bathy_depth', 'cage_depth', 'bathy_min', 'bathy_max', 'bathy_mean']]
         ],
        axis=1)

    # Convert back
    facilities_depth.to_crs(f'EPSG:{CRS_DICT["mapping"]}', inplace=True)
    return facilities_depth


def dedup_cages_in_overlap_years_with_white_space(
        cages: gpd.GeoDataFrame,
        image_boxes: gpd.GeoDataFrame,
        pass_map: Callable,
        year_selection: str
) -> tuple[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
    """
    Removes cages in duplicate areas that are covered by multiple years within the same pass,
    using the geometries of the image boxes (i.e., accounting for whitespace in the images)
    Args:
    :param year_selection: ['random', 'min', 'max']
    :param cages: a GeoDataFrame of cages.
    :param image_boxes: a GeoDataFrame of image bboxes
    :param pass_map: function to convert years to periods
    Returns:
        a GeoDataFrame of deduplicated cages.
    """

    def tile_coverage(tile_box_df: gpd.GeoDataFrame) -> List[tuple[int, polygon.Polygon]]:
        """
        Uses the image boxes in the tile_box_df gdf to uniquely cover a specific tile, such that
        areas from the different image boxes do not overlap. It returns a list with the image box
        index and the modified geometry. The tile is covered exactly in the order of the gdf, such
        that the first image box's geometry is fully used, and the rest use only the non-overlapping
        areas.
        :param tile_box_df: gpd of image boxes for a specific tile key (defined by bbox_ind, x_off, y_off)
        :return: list of ['image_box_index', modified geometry]
        """
        # Get existing coverage
        existing_coverage = tile_box_df.iloc[[0]].copy()
        existing_coverage_dict = existing_coverage.iloc[0]

        output = [(existing_coverage_dict['image_box_index'], existing_coverage_dict['geometry'])]
        for i in range(1, len(tile_box_df)):
            # Get row
            box_row = tile_box_df.iloc[[i]]
            new_row = gpd.overlay(box_row, existing_coverage, how='difference')

            # Save
            if len(new_row) > 0:
                output.append((new_row.iloc[0]['image_box_index'], new_row.iloc[0]['geometry']))

            # Update coverage
            existing_coverage = pd.concat([existing_coverage, new_row])
            existing_coverage = gpd.GeoDataFrame(existing_coverage, crs=tile_box_df.crs)
        return output

    def img_box_allocation_in_tile(
            tile_row: pd.Series, selection: str, tiles_img_boxes: gpd.GeoDataFrame,
            tiles_cages: gpd.GeoDataFrame = None) -> List[tuple[int, polygon.Polygon]]:
        """
        Selects the image boxes to be used to uniquely cover a tile (as defined by bbox_ind, x_off and y_off). If
        random, it uses a random ordering of all available image boxes in the tile. If min (max), it uses the spatially
        unique permutation of image boxes that result in the min (max) cage area for the tile.
        :param tile_row: Series including the tile_key (bbox_ind-x_off-y_off) for which the allocation will be performed
        :param selection: One of ['min', 'max', 'random']
        :param tiles_img_boxes: gdf of image boxes
        :param tiles_cages: gdf of cages
        :return: list of ['image_box_index', modified geometry] used to uniquely cover the tile under the selected
        allocation.
        """
        # Get tile key
        row_tile_key = tile_row['tile_key']

        # Get image boxes that fall within the tile
        tile_boxes = tiles_img_boxes.loc[tiles_img_boxes['tile_key'] == row_tile_key]

        output = None
        if selection == 'random':
            # Random shuffle
            tile_boxes = tile_boxes.sample(frac=1)
            output = tile_coverage(tile_box_df=tile_boxes)
        else:
            assert tiles_cages is not None

            # Get all possible permutations of the rows
            perms = list(range(len(tile_boxes)))
            perms = list(iter_permutations(perms))

            # Set up min and max
            min_cage_area = np.Inf
            max_cage_area = 0
            min_output = None
            max_output = None

            for perm in perms:
                tile_boxes_perm = tile_boxes.iloc[list(perm)]
                perm_output = tile_coverage(tile_box_df=tile_boxes_perm)

                # Convert perm output list to dataframe
                perm_df = pd.DataFrame.from_dict({'selected_image_boxes': [perm_output]})
                perm_df = parse_output_list_to_gdf(
                    df=perm_df, explode_col='selected_image_boxes')

                # Compute cage area
                cage_area = tiles_cages.loc[tiles_cages['tile_key'] == row_tile_key]
                if len(cage_area) == 0:
                    cage_area = 0
                else:
                    cage_area = filter_cages_to_image_boxes(img_box_df=perm_df, cage_df=cage_area)
                    cage_area = cage_area['area'].sum()

                if cage_area >= max_cage_area:
                    max_output = perm_output
                    max_cage_area = cage_area
                if cage_area < min_cage_area:
                    min_output = perm_output
                    min_cage_area = cage_area

            if selection == 'min':
                output = min_output
            elif selection == 'max':
                output = max_output

        return output

    def parse_output_list_to_gdf(df: pd.DataFrame, explode_col: str) -> gpd.GeoDataFrame:
        """
        Converts the list of [image_box_idx, modified geometry] generated by the function img_box_allocation_in_tile
        into a gdf.
        :param df: df including the output list in column explode_col
        :param explode_col:
        :return: a gdf containing the columns image box index and geometry (modified geometry of the image box
        following the allocation to a tile)
        """
        df = df.explode(explode_col)

        df['image_box_index'], df['geometry'] = zip(*df[explode_col])
        df['image_box_index'] = df['image_box_index'].astype(int)

        df = gpd.GeoDataFrame(df, crs="EPSG:3857")
        return df

    def filter_cages_to_image_boxes(img_box_df: gpd.GeoDataFrame, cage_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Uses the gdf produced by the function parse_output_list_to_gdf containing image_box_idx and the modified
        image box geometries to filter a gdf of cages to only those that fall within the modified image boxes.
        :param img_box_df: gdf of modified image boxes
        :param cage_df: gdf of cages
        :return: filtered gdf of cages
        """
        # Check cage data frame
        if len(cage_df) == 0:
            return gpd.GeoDataFrame()

        # Cages
        in_cages = cage_df.merge(
            img_box_df[['image_box_index', 'geometry']],
            how='left', validate='many_to_one', on='image_box_index')

        # Note that this bool also captures observations where a match for the image box could not be found so
        # we don't have to include a separate check for this
        in_cages['in_imgbox'] = in_cages.apply(
            lambda row: row['geometry_x'].intersects(row['geometry_y']), axis=1)
        in_cages = in_cages.loc[in_cages['in_imgbox'] == True].copy()

        in_cages['geometry'] = in_cages['geometry_x']
        in_cages.drop(['geometry_y', 'geometry_x'], axis=1, inplace=True)
        in_cages = gpd.GeoDataFrame(in_cages, crs=cage_df.crs)
        return in_cages

    assert year_selection in ['min', 'max', 'random']

    image_boxes = image_boxes.to_crs("EPSG:3857")
    cages = cages.to_crs("EPSG:3857")

    # Assign period to image_boxes
    image_boxes['year'] = image_boxes['year'].astype(int)
    image_boxes['pass'] = image_boxes['year'].map(pass_map)
    image_boxes['year'] = image_boxes['year'].astype(str)
    cages['pass'] = cages['year'].map(pass_map)

    # Create an image box index
    image_boxes.reset_index(drop=True, inplace=True)
    image_boxes['image_box_index'] = image_boxes.index

    # Get year, bbox ind, and offsets for each cage so we can merge on these
    cages['year_img'] = cages['year'].astype(str)
    cages.drop('year', axis=1, inplace=True)
    cages.reset_index(drop=True, inplace=True)
    cages = pd.concat([
        cages,
        pd.DataFrame.from_records(cages['image'].apply(lambda file: generate_image_specs_from_file_name(file)))],
        axis=1)
    assert (cages['year_img'] != cages['year']).sum() == 0

    # Generate tile key in image boxes
    image_boxes['tile_key'] = image_boxes.apply(
        lambda row: f"{row['bbox_ind']}-{row['x_offset']}-{row['y_offset']}", axis=1)

    # Add img_box_index to cage data frame for merging
    for c in ['bbox_ind', 'x_offset', 'y_offset']:
        image_boxes['bbox_ind'] = image_boxes['bbox_ind'].astype(str)
    cages = cages.merge(
        image_boxes[['bbox_ind', 'x_offset', 'y_offset', 'year', 'image_box_index']],
        on=['bbox_ind', 'x_offset', 'y_offset', 'year'], validate='many_to_one', how='left'
    )

    # Create the tile key
    tile_key = image_boxes.copy()
    tile_key = tile_key.groupby(['pass', 'bbox_ind', 'x_offset', 'y_offset', 'tile_key'])[
        'image_file'].count().reset_index()

    deduped_cages = pd.DataFrame()
    annual_coverage = {}
    for selected_pass in image_boxes['pass'].unique():
        # Filter to period
        tile_key_pass = tile_key.loc[tile_key['pass'] == selected_pass].copy()
        image_boxes_pass = image_boxes.loc[image_boxes['pass'] == selected_pass].copy()
        cages_pass = cages.loc[cages['pass'] == selected_pass].copy()

        # Determine which tiles have unique and which have multiple imagery
        unique_tiles_pass = tile_key_pass.loc[tile_key_pass['image_file'] == 1]
        nonunique_tiles_pass = tile_key_pass.loc[tile_key_pass['image_file'] > 1]
        assert len(unique_tiles_pass) + len(nonunique_tiles_pass) == len(tile_key_pass)

        # For the tiles that have a single image, we just use the one image box
        dedup_image_boxes_pass = image_boxes_pass.loc[
            image_boxes_pass['tile_key'].isin(unique_tiles_pass['tile_key'].unique())]
        dedup_image_boxes_pass = dedup_image_boxes_pass[['image_box_index', 'geometry']]

        # For the tiles with multiple imagery, we select the image boxes to use
        cages_pass['tile_key'] = cages_pass.apply(
            lambda row: f"{row['bbox_ind']}-{row['x_offset']}-{row['y_offset']}", axis=1)
        nonunique_tiles_pass_dd = dd.from_pandas(nonunique_tiles_pass, npartitions=4)
        with ProgressBar():
            result = nonunique_tiles_pass_dd.apply(
                img_box_allocation_in_tile, axis=1, selection=year_selection, tiles_cages=cages_pass,
                tiles_img_boxes=image_boxes_pass, meta=('select_image_boxes', object))
            nonunique_tiles_pass['select_image_boxes'] = result.compute()

        # Get gdf of image box indices and their modified geometries
        nonunique_image_boxes_pass = parse_output_list_to_gdf(
            df=nonunique_tiles_pass, explode_col='select_image_boxes')
        nonunique_image_boxes_pass = nonunique_image_boxes_pass[list(dedup_image_boxes_pass.columns)]
        dedup_image_boxes_pass = pd.concat([dedup_image_boxes_pass, nonunique_image_boxes_pass])

        # Lastly, we filter the cages to the selected image boxes and their modified geometries
        dedup_cages_pass = filter_cages_to_image_boxes(img_box_df=dedup_image_boxes_pass, cage_df=cages_pass)

        deduped_cages = pd.concat([deduped_cages, dedup_cages_pass])
        annual_coverage[selected_pass] = dedup_image_boxes_pass

    return deduped_cages, annual_coverage


class AquaFacility:
    def __init__(
            self,
            main_dir: str,
            selected_map: Callable,
            image_selection: str,
            confidence_threshold: float,
            distance_threshold: float,
            min_cluster_size: float,
            time_group: str
    ):

        # Save parameters
        self.main_dir = main_dir
        self.selected_map = selected_map
        self.confidence_threshold = confidence_threshold
        self.image_selection = image_selection
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.time_group = time_group

        # Obtain the image boxes
        rmblank_image_boxes = load_final_image_boxes(main_dir=main_dir)
        download_bboxes = load_download_bboxes(os.path.join(main_dir, "data/wanted_bboxes.csv"))
        dedup_boxes = deduplicate_download_boxes(
            download_bboxes, path=os.path.join(main_dir, "data/wanted_bboxes_dedup.csv"))

        # Get CF labels. Note that this is not cage area but bounding box area
        print('[INFO] Using CF labels to filter by maximum cage area')
        labels = load_cf_labels().rename(columns={"fn": "image"})
        labels['bbox_ind'] = labels['image'].apply(lambda i: i.split('_')[1])
        labels = deduplicate_gdf_with_bboxes(dedup_boxes=dedup_boxes, gdf_todedup=labels, path=None)

        # Note that the get_cage_min_and_max_areas function converts to the right CRS for area calculation
        max_cage_areas = get_cage_min_and_max_areas(labels)
        max_cage_areas.index = max_cage_areas.index.map({'circle_cage': 'circle_farm', 'square_cage': 'square_farm'})

        # Load predictions and drop those below the maximum cage areas as defined by CF labels
        preds = gpd.read_file(os.path.join(main_dir, "output/ocean_detections.geojson"))
        preds['farm_type'] = preds['type']
        preds = preds[preds.to_crs(f"EPSG:{CRS_DICT['area']}").area < preds['type'].map(max_cage_areas)]
        preds['pass'] = preds['year'].map(self.selected_map)

        # Filter predictions by confidence threshold
        preds = preds[preds['det_conf'] >= confidence_threshold]
        preds.reset_index(drop=True, inplace=True)
        preds['index'] = preds.index   # This will be the cage's unique identifier.

        # For uncertainty: need to save predictions and image boxes
        self.preds = preds
        self.rmblank_image_boxes = rmblank_image_boxes
        self.min_max_selection = None

        # Deduplicate predictions: this handles both spatial duplication from the bounding boxes,
        # and duplication arising from images from different years covering the same location
        print(f'[INFO] De-duplicating cages within time periods using selection {image_selection}')
        facility_cages, annual_coverage, final_facilities = self.deduplicate_and_cluster(year_selection=image_selection)
        print(f'[INFO] {len(facility_cages)} cages clustered into {len(final_facilities)} facilities')

        assert final_facilities.crs == f'EPSG:{CRS_DICT["mapping"]}'
        assert facility_cages.crs == f'EPSG:{CRS_DICT["mapping"]}'
        self.facility_cages = facility_cages.copy()   # This is the final subset of preds that belongs to the facilities.
        self.final_facilities = final_facilities.copy()
        self.annual_coverage = annual_coverage

    def deduplicate_and_cluster(self, year_selection):
        cages, annual_coverage = dedup_cages_in_overlap_years_with_white_space(
            cages=self.preds, image_boxes=self.rmblank_image_boxes, pass_map=self.selected_map, year_selection=year_selection)

        # Cluster cages into facilities
        facilities = DBSCAN_cluster(
            cages.to_crs(f"EPSG:{CRS_DICT['area']}"),
            facilities_path='',
            distance_threshold=self.distance_threshold,
            amnt_min_clusters=self.min_cluster_size,
            include_area=True,
            save=False,
            return_detections=False,
            cluster_variable=self.time_group
        )
        assert facilities.crs == f'EPSG:{CRS_DICT["mapping"]}'
        return cages, annual_coverage, facilities

    def compute_min_max_cages(self):
        # To incorporate uncertainty from de-duplication, we do the same for minimum and maximum
        # year selection.
        final_facilities = self.final_facilities.copy()
        if 'cage_ids_max' in final_facilities.columns:
            return

        print('[INFO] Performing deduplication using minimum and maximum cage area selection for uncertainty measures')
        min_max_selection = {'min': None, 'max': None}
        for year_selection in ['min', 'max']:
            _, _, facilities_selection = self.deduplicate_and_cluster(year_selection=year_selection)
            min_max_selection[year_selection] = facilities_selection

        # Merge to facilities_df -> we need to add two columns: ['cage_ids_min', 'cage_ids_max']
        for year_selection in ['min', 'max']:
            facilities_selection = min_max_selection[year_selection].copy()
            final_facilities_gdf = self.final_facilities.copy()

            # Re-define cage geometries (consolidate all square and circle cages)
            final_facilities_gdf['all_cages'] = final_facilities_gdf.apply(
                lambda row: row['square_farm_geoms'].buffer(0).union(row['circle_farm_geoms'].buffer(0)), axis=1)

            facilities_selection['all_cages'] = facilities_selection.apply(
                lambda row: row['square_farm_geoms'].buffer(0).union(row['circle_farm_geoms'].buffer(0)), axis=1)

            final_facilities_gdf.set_geometry('all_cages', crs=final_facilities_gdf.crs, inplace=True)
            facilities_selection.set_geometry('all_cages', crs=facilities_selection.crs, inplace=True)

            final_facilities_gdf = final_facilities_gdf[['facility_index', 'all_cages', 'cage_ids', 'pass']]
            facilities_selection = facilities_selection[['facility_index', 'all_cages', 'cage_ids', 'pass']]

            # Check that we have unique indices for the facilities (these are automatically generated
            # in the DBSCAN_cluster function)
            assert len(final_facilities_gdf['facility_index'].unique()) == len(final_facilities_gdf)
            assert len(facilities_selection['facility_index'].unique()) == len(facilities_selection)

            # Spatial join and keep matches from the same pass
            final_facilities_gdf.to_crs(f"EPSG:{CRS_DICT['area']}", inplace=True)
            facilities_selection.to_crs(f"EPSG:{CRS_DICT['area']}", inplace=True)
            final_facilities_gdf = final_facilities_gdf.sjoin(facilities_selection, predicate='intersects')
            final_facilities_gdf = final_facilities_gdf.loc[
                final_facilities_gdf['pass_left'] == final_facilities_gdf['pass_right']]

            # Add geometry info
            final_facilities_gdf = final_facilities_gdf.merge(
                facilities_selection[['facility_index', 'all_cages']],
                how='left', left_on='facility_index_right', right_on='facility_index', validate='many_to_one')

            # Get overlap area and select highest overlap facility
            final_facilities_gdf['overlap'] = final_facilities_gdf.apply(
                lambda row: row['all_cages_x'].intersection(row['all_cages_y']).area / row['all_cages_x'].area, axis=1)
            final_facilities_gdf.sort_values('overlap', ascending=False, inplace=True)
            final_facilities_gdf.drop_duplicates(subset=['facility_index_left'], keep='first', inplace=True)

            # Save to final facilities
            final_facilities_gdf.rename(columns={'cage_ids_right': f'cage_ids_{year_selection}'}, inplace=True)
            final_facilities = final_facilities.merge(
                final_facilities_gdf[['facility_index_left', f'cage_ids_{year_selection}']], left_on='facility_index',
                right_on='facility_index_left',
                how='left', validate='one_to_one')
            final_facilities.drop('facility_index_left', axis=1, inplace=True)

        # Create empty lists for the facilities with no cages under the minimum year selection
        final_facilities['cage_ids_min'] = final_facilities['cage_ids_min'].fillna("").apply(list)

        assert final_facilities.crs == f'EPSG:{CRS_DICT["mapping"]}'
        self.final_facilities = final_facilities
        self.min_max_selection = min_max_selection

    def add_depth(self, min_cage_threshold: float, default_cage_depth: float, bathymetry_statistic: str):
        """
        Incorporate bathymetry data to estimate facility-level cage depth. We use a minimum depth threshold and a
        default cage depth for facilities that have no bathymetry data available. The bathymetry data is downloaded
        using the bathymetry_data.py script
        :param min_cage_threshold: (meters)
        :param default_cage_depth: (meters)
        """
        facilities_gdf = self.final_facilities
        if 'cage_depth' not in facilities_gdf.columns:
            facilities_gdf = add_facility_depth(
                facility_df=facilities_gdf,
                bathymetry_path=os.path.join(self.main_dir, 'data/bathymetry/EMOD_2022.tif'),
                min_cage_threshold=min_cage_threshold,
                default_cage_depth=default_cage_depth,
                bathymetry_statistic=bathymetry_statistic
            )
            self.final_facilities = facilities_gdf
            assert facilities_gdf.crs == f'EPSG:{CRS_DICT["mapping"]}'

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)


def load_AquaFacility(filename=None, *args, **kwargs):
    """
    Loads an existing AquaFacility at the given path or generates a new instance.
    @param filename: pkl path to the saved facility, if loading an existing facility
    @param args:
    @param kwargs:
    @return:
    """
    if filename:
        print('[INFO] Loading saved AquaFacility from file...')
        with open(filename, 'rb') as pickle_file:
            inst = pickle.load(pickle_file)
    else:
        inst = AquaFacility(*args, **kwargs)
    return inst


def modify_cage_list_using_geometry(
        cage_ids: List[int],
        bounds: shapely.geometry.polygon,
        cage_df: gpd.GeoDataFrame,
        sbound: str) -> List[int]:
    """
    Modifies a list of cages according to whether they fall inside or outside a given geometry.
    :param cage_ids: list of cage ids. The ids must match the index column in cage_df
    :param bounds: The geometry against which we'll check where the cages are
    :param cage_df: gdf of cage geometries
    :param sbound: Whether to keep cages that are inside or outside the geometry
    :return: modified cage_ids list
    """
    assert sbound in ['inside', 'outside']
    new_cage_ids = []
    for cage_id in cage_ids:
        # Get the cage's geometry
        cage_geom = cage_df.loc[cage_df['index'] == cage_id]
        if len(cage_geom) == 1:
            cage_geom = cage_geom.iloc[0]['geometry']
            intersects = cage_geom.intersects(bounds)
            keep = False
            if (sbound == 'inside' and intersects) or (sbound == 'outside' and not intersects):
                keep = True
            if keep:
                new_cage_ids.append(cage_id)
        else:
            print('[WARNING] Multiple or zero cage matches for cage index.')
    return new_cage_ids


def compute_complete_period_tonnage_estimates(
        Facility: AquaFacility,
        current_period: str,
        compare_period: str,
        model_error_distributions: pd.DataFrame,
        min_cage_threshold: float,
        period_factor_table: pd.DataFrame,
        depth_dist_mixture_param: float,
        K: int = 10_000) -> pd.DataFrame:
    """
    Computes tonnage estimates (and uncertainty) for a current period, including cages from compare_period that
    fall outside the spatial coverage of the imagery of the current period. That is, this function accounts for the
    missing imagery in a period and imputes missing facilities using another period.
    :param Facility: AquaFacility object
    :param current_period: The period for which we'll compute complete tonnage estimates
    :param compare_period: The period used to impute the facilities missing from the current period
    :param model_error_distributions:
    :param period_factor_table:
    :param min_cage_threshold:
    :param K: Number of bootstrap iterations
    :return:
    """

    bounded_facilities = Facility.final_facilities.copy()
    preds_df = Facility.preds.copy()
    pass_bounds = Facility.annual_coverage[current_period].copy()
    pass_bounds = pass_bounds['geometry'].unary_union

    # Use facilities from current period and compare period
    bounded_facilities = bounded_facilities.loc[bounded_facilities['pass'].isin([compare_period, current_period])]

    # For each cage in this facility df from the compare period, check if against the bounds to see if we should keep it
    for cage_id_col in ['cage_ids', 'cage_ids_max', 'cage_ids_min']:
        bounded_facilities[cage_id_col] = bounded_facilities.apply(
            lambda row: row[cage_id_col] if row['pass'] == current_period else modify_cage_list_using_geometry(
                cage_ids=row[cage_id_col], bounds=pass_bounds, cage_df=preds_df, sbound='outside'), axis=1)

    # Drop facilities with no cages
    bounded_facilities['length'] = bounded_facilities['cage_ids_min'].apply(lambda l: len(l))
    bounded_facilities = bounded_facilities.loc[bounded_facilities['length'] > 0]
    print(f'Number of total facilities: {len(bounded_facilities)}')
    print(
        f'Number of added facilities from compare period: {len(bounded_facilities.loc[bounded_facilities["pass"] == compare_period])}')

    # At this point we need to update the pass of the facilities from the compare period
    bounded_facilities['pass'] = current_period

    if len(bounded_facilities) == 0:
        estimates = pd.DataFrame.from_dict({'tonnage': [0]})
    else:
        # Compute tonnage estimates
        estimates = compute_facility_tonnage_estimates(
            facility_df=bounded_facilities,
            period_factor_table=period_factor_table,
            period_var=Facility.time_group,
            min_cage_threshold=min_cage_threshold,
            preds_df=preds_df,
            model_error_distributions=model_error_distributions,
            depth_dist_mixture_param=depth_dist_mixture_param,
            K=K
        )

    return estimates.loc[[current_period]]


class CF_Facility(AquaFacility):
    """
    AquaFacility object for the human-annotated cages.
    """
    def __init__(self, final_facilities, preds, cages, annual_coverage, selected_map, distance_threshold,
                 min_cluster_size, rmblank_image_boxes, main_dir, image_selection):
        self.main_dir = main_dir
        self.selected_map = selected_map
        self.confidence_threshold = 0
        self.image_selection = image_selection
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.time_group = 'pass'

        self.preds = preds
        self.rmblank_image_boxes = rmblank_image_boxes
        self.min_max_selection = None

        self.facility_cages = cages
        self.final_facilities = final_facilities
        self.annual_coverage = annual_coverage
