"""
Generates the French Mediterranean map of known vs additional facility locations: Figure 4 ("Marine finfish
aquaculture production locations in the French Mediterranean") of the manuscript
"""
import os
from typing import Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, box
from mpl_toolkits.basemap import Basemap

from src.utils import (
    load_cf_labels, load_download_bboxes, map_year_to_image_pass_opt2, load_Trujillo_locations_deduped
)
from src.utils_tonnage import load_AquaFacility
import src.file_utils as file_utils


def define_Trujillo_locations() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Collects the locations identified by Trujillo et al. in the French Mediterranean, as well as 1km boxes
    around these locations.
    @return:
    """
    # Unduplicated set of Trujillo locations
    root_dir = file_utils.get_root_path()
    Trujillo_locations = load_Trujillo_locations_deduped(path=f'{root_dir}/data/aquaculture_med.csv')
    box_size = 1000

    # Define 1 km boxes
    Trujillo_boxes = Trujillo_locations.copy()
    Trujillo_boxes.to_crs('EPSG:3035', inplace=True)
    Trujillo_boxes['1km_box'] = Trujillo_boxes['geometry'].apply(
        lambda p: box(p.x - box_size, p.y - box_size, p.x + box_size, p.y + box_size))
    Trujillo_boxes.set_geometry('1km_box', inplace=True, crs=Trujillo_boxes.crs)
    Trujillo_boxes = Trujillo_boxes[['name', '1km_box']]
    Trujillo_boxes.to_crs('EPSG:3857', inplace=True)

    # Add boxes back to Trujillo_locations
    Trujillo_locations = Trujillo_locations.merge(Trujillo_boxes, on='name', validate='one_to_one')

    # French locations
    download_bboxes = load_download_bboxes()
    xmin, ymin, xmax, ymax = download_bboxes.total_bounds
    Trujillo_locations_france = Trujillo_locations.cx[xmin:xmax, ymin:ymax].copy()

    Trujillo_facilities_france = Trujillo_locations_france.groupby(['lat', 'lon'])['num_cages'].sum().reset_index()
    Trujillo_facilities_france['trujillo_facility_index'] = Trujillo_facilities_france.index
    Trujillo_facilities_france = Trujillo_facilities_france[['trujillo_facility_index', 'lat', 'lon', 'num_cages']]

    Trujillo_locations_france = Trujillo_locations_france.merge(
        Trujillo_facilities_france[['trujillo_facility_index', 'lat', 'lon']], how='left', on=['lat', 'lon'],
        validate='many_to_one')

    Trujillo_facility_boxes = Trujillo_locations_france[['1km_box', 'trujillo_facility_index']].drop_duplicates()
    Trujillo_facility_boxes = gpd.GeoDataFrame(
        Trujillo_facility_boxes, geometry='1km_box', crs=Trujillo_locations_france.crs)
    return Trujillo_facility_boxes, Trujillo_facilities_france


def get_true_facilities() -> gpd.GeoDataFrame:
    """
    Returns the positive instances of mariculture cage clusters identified by our prediction model, using
    the human-annotated labels.
    @return:
    """
    root_dir = file_utils.get_root_path()

    # Load labels and our predictions
    cf_labels = load_cf_labels().rename(columns={"fn": "image"})
    PredictionFacility = load_AquaFacility(
        filename=f'{root_dir}/output/Facilities/AQ_tunedfacility.pkl',
        main_dir=str(root_dir), selected_map=None, image_selection=None, confidence_threshold=None,
        distance_threshold=None, min_cluster_size=None, time_group=None
    )
    our_facilities = PredictionFacility.final_facilities.copy()
    assert cf_labels.crs == our_facilities.crs

    # Get only true positive clusters
    true_facilities = our_facilities.copy()
    true_facilities['all_cages'] = true_facilities.apply(
        lambda row: row['square_farm_geoms'].buffer(0).union(row['circle_farm_geoms'].buffer(0)), axis=1)
    true_facilities['bounds'] = true_facilities['all_cages'].apply(lambda all_cages: box(*all_cages.bounds))
    true_facilities = true_facilities.set_geometry('bounds', crs=true_facilities.crs)
    true_facilities = true_facilities.sjoin(cf_labels, how='left')
    true_facilities = true_facilities[['bounds', 'pass', 'facility_index', 'year', 'cage_ids']]
    true_facilities['cf_pass'] = true_facilities['year'].map(map_year_to_image_pass_opt2)
    true_facilities = true_facilities.loc[(true_facilities['pass'] == true_facilities['cf_pass'])].copy()
    true_facilities.drop_duplicates('facility_index', inplace=True)

    return true_facilities


def count_unique_locations(fac_df: gpd.GeoDataFrame) -> int:
    """
    Counts the number of unique cage clusters in a GeoDataFrame
    @param fac_df:
    @return:
    """
    unique_facs = fac_df[['facility_index', 'bounds', 'pass']].sjoin(fac_df[['facility_index', 'bounds', 'pass']])

    unique_facs = unique_facs.groupby('facility_index_left')['facility_index_right'].apply(
        lambda x: list(np.unique(x))).reset_index()
    unique_facilities = []
    nonunique_facilities = []
    for i in range(len(unique_facs)):
        row = unique_facs.iloc[i]
        if row['facility_index_left'] not in nonunique_facilities:
            unique_facilities.append(row['facility_index_left'])
            nonunique_facilities.extend(row['facility_index_right'])
    return len(unique_facilities)


def classify_our_facilities(
        true_facilities: gpd.GeoDataFrame,
        Trujillo_facility_boxes: gpd.GeoDataFrame,
        Trujillo_facilities_france: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Determines which of the cage clusters located by our prediction model belong to the known facilities
    identified by Trujillo et al., and which are new/unknown production locations.
    @param true_facilities:
    @param Trujillo_facility_boxes:
    @param Trujillo_facilities_france:
    @return:
    """
    assert true_facilities.crs == Trujillo_facility_boxes.crs

    # Get Trujillo locations
    Trujillo_facilities_france = gpd.GeoDataFrame(
        Trujillo_facilities_france,
        geometry=gpd.points_from_xy(Trujillo_facilities_france.lon, Trujillo_facilities_france.lat),
        crs='EPSG:4326'
    )
    Trujillo_facilities_france.to_crs(true_facilities.crs, inplace=True)

    # Define number of cages for our predicted facilities
    true_facilities['num_cages'] = true_facilities['cage_ids'].apply(lambda x: int(len(x)))

    # 1. Define additional farms during the Trujillo study period (2002-2010) outside of their bounding boxes
    add_trujillo = true_facilities.loc[
        true_facilities['pass'].isin(['2000-2004', '2005-2009', '2010-2012'])].copy()

    # Our facilities outside of the 1km boxes
    add_trujillo = add_trujillo.sjoin(Trujillo_facility_boxes, how='left', predicate='intersects')
    add_trujillo = add_trujillo.loc[add_trujillo['trujillo_facility_index'].isna()]
    add_trujillo.drop_duplicates('facility_index', inplace=True)
    add_trujillo['type'] = 'Additional facility'
    num_unique_trujillo = count_unique_locations(fac_df=add_trujillo)
    print(f'Number of unique additional facilities during Trujillo period: {num_unique_trujillo}')

    add_trujillo['geometry'] = add_trujillo['bounds'].centroid
    add_trujillo.set_geometry('geometry', crs=add_trujillo.crs, inplace=True)

    combined_facilities = add_trujillo.copy()
    for time_period in ['2000-2004', '2005-2009', '2010-2012']:
        Trujillo_facilities_france_time = Trujillo_facilities_france.copy()
        Trujillo_facilities_france_time['pass'] = time_period
        Trujillo_facilities_france_time['type'] = 'Known facility'
        combined_facilities = pd.concat([
            combined_facilities[['type', 'geometry', 'pass', 'num_cages']],
            Trujillo_facilities_france_time[['type', 'geometry', 'pass', 'num_cages']]
        ])

    # 2. Farms post-Trujillo study period
    all_posttrujillo = true_facilities.loc[
        true_facilities['pass'].isin(['2013-2015', '2016-2018', '2019-2021'])].copy()
    all_posttrujillo = all_posttrujillo.sjoin(Trujillo_facility_boxes, how='left', predicate='intersects')
    all_posttrujillo.sort_values('trujillo_facility_index', ascending=False, inplace=True)
    all_posttrujillo.drop_duplicates('facility_index', inplace=True)
    all_posttrujillo['type'] = all_posttrujillo['trujillo_facility_index'].apply(
        lambda i: 'Additional facility' if pd.isnull(i) else 'Known facility')

    num_unique_posttrujillo = count_unique_locations(
        fac_df=all_posttrujillo.loc[all_posttrujillo['type'] == 'Additional facility'])
    print(f'Number of unique additional facilities post Trujillo period: {num_unique_posttrujillo}')

    all_posttrujillo['geometry'] = all_posttrujillo['bounds'].centroid
    all_posttrujillo.set_geometry('geometry', crs=all_posttrujillo.crs, inplace=True)

    combined_facilities = pd.concat([
        combined_facilities[['type', 'geometry', 'pass', 'num_cages']],
        all_posttrujillo[['type', 'geometry', 'pass', 'num_cages']]
    ])
    return combined_facilities


def plot_map(combined_facilities: gpd.GeoDataFrame) -> None:
    """
    Plots the French Mediterranean map including known and additional cage clusters, over time.
    @param combined_facilities:
    """
    root_dir = file_utils.get_root_path()

    # Create cage buckets and change CRS for mapping
    combined_facilities['cage_bin'] = pd.cut(combined_facilities['num_cages'], [0, 50, 100, 500]).astype(str)
    combined_facilities.to_crs('EPSG:4326', inplace=True)

    # Map set up
    download_bboxes = load_download_bboxes()
    xmin, ymin, xmax, ymax = download_bboxes.to_crs('EPSG:3035').buffer(20000).to_crs('EPSG:4326').total_bounds
    size_fn = {'(0, 50]': 10, '(50, 100]': 30, '(100, 500]': 70}
    color_fn = {'Additional facility': 'blue', 'Known facility': 'red'}

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10.5, 10))
    (ax1, ax2), (ax3, ax4), (ax5, ax6) = axs
    for time_period, ax in zip(['2000-2004', '2005-2009', '2010-2012', '2013-2015', '2016-2018', '2019-2021'],
                               [ax1, ax2, ax3, ax4, ax5, ax6]):
        m = Basemap(
            projection='merc',
            llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
            resolution='h', ax=ax)
        m.drawcountries()
        m.fillcontinents()

        # Plot additional facilities for the time period
        period_facilities = combined_facilities.loc[combined_facilities['pass'] == time_period].copy()

        period_lons = period_facilities['geometry'].map(lambda p: p.x)
        period_lats = period_facilities['geometry'].map(lambda p: p.y)
        x_period, y_period = m(period_lons, period_lats)
        our = m.scatter(x_period, y_period, marker='o', c=period_facilities['type'].map(color_fn), alpha=0.6,
                        s=period_facilities.cage_bin.map(size_fn), label=period_facilities.type, linewidth=0)
        ax.set_title(time_period)
    fig.tight_layout()
    plt.savefig(f'{root_dir}/output/paper_figures/facilities_in_time.png', dpi=300, bbox_inches='tight')


def plot_map_single_period(combined_facilities: gpd.GeoDataFrame) -> None:
    """
    Plots the French Mediterranean map including known and additional cage clusters, collapsed in time.
    @param combined_facilities:
    """
    root_dir = file_utils.get_root_path()

    # Create cage buckets and change CRS for mapping
    combined_facilities['cage_bin'] = pd.cut(combined_facilities['num_cages'], [0, 50, 100, 500]).astype(str)
    combined_facilities.to_crs('EPSG:4326', inplace=True)

    # Map set up
    download_bboxes = load_download_bboxes()
    xmin, ymin, xmax, ymax = download_bboxes.to_crs('EPSG:3035').buffer(20000).to_crs('EPSG:4326').total_bounds
    size_fn = {
        '(0, 50]': 20,
        '(50, 100]': 90,
        '(100, 500]': 200
    }
    color_fn = {'Additional facility': 'blue', 'Known facility': 'red'}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3.5))
    m = Basemap(
        projection='merc',
        llcrnrlon=xmin, llcrnrlat=ymin, urcrnrlon=xmax, urcrnrlat=ymax,
        resolution='h', ax=ax)
    m.drawcountries()
    m.fillcontinents()

    # Collapse into a single panel:
    # * We show the original Trujillo facilities
    collapsed_facilities = combined_facilities.loc[((
        combined_facilities['type'] == 'Known facility') & (combined_facilities['pass'] == '2000-2004') | (
        (combined_facilities['type'] == 'Additional facility')))]

    period_lons = collapsed_facilities['geometry'].map(lambda p: p.x)
    period_lats = collapsed_facilities['geometry'].map(lambda p: p.y)
    x_period, y_period = m(period_lons, period_lats)
    our = m.scatter(x_period, y_period, marker='o', c=collapsed_facilities['type'].map(color_fn), alpha=0.6,
                    s=collapsed_facilities.cage_bin.map(size_fn), label=collapsed_facilities.type, linewidth=0)
    fig.tight_layout()
    plt.savefig(f'{root_dir}/output/paper_figures/facilities.pdf', dpi=300, bbox_inches='tight', format='pdf')


if __name__ == '__main__':
    root_dir = file_utils.get_root_path()
    Trujillo_fac_boxes, Trujillo_fac_france = define_Trujillo_locations()
    true_facs = get_true_facilities()
    combined_facs = classify_our_facilities(
        true_facilities=true_facs, Trujillo_facility_boxes=Trujillo_fac_boxes,
        Trujillo_facilities_france=Trujillo_fac_france)

    os.makedirs(f'{root_dir}/output/paper_figures', exist_ok=True)
    plot_map(combined_facilities=combined_facs)
    plot_map_single_period(combined_facilities=combined_facs)
