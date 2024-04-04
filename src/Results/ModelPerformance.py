"""
Computes stats on the model's performance (precision and recall), and generates Figure 3 ("Performance
on the French Mediterranean coast") of the manuscript.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils import stylize_axes, map_year_to_image_pass_opt2
from src.cluster_facilities import DBSCAN_cluster
from src.get_kfold_cluster_performance import (
    get_tp, load_datasets_for_model_evaluation, get_stats_total, get_bucket_info_table
)
import src.file_utils as file_utils


def get_sample_stats_at_thresholds(
    labels,
    sample_predictions,
    thresholds: np.linspace
) -> pd.DataFrame:

    stats_dicts = []
    for threshold in tqdm(thresholds):
        threshold_predictions = sample_predictions.loc[sample_predictions['det_conf'] >= threshold].copy()

        stats = get_stats_total(labels=labels, preds=threshold_predictions)
        stats['threshold'] = threshold
        stats_dicts.append(stats)

    return pd.DataFrame.from_records(stats_dicts)


def plot_new_figure(all_stats, ocean_stats, cluster_stats):
    root_dir = file_utils.get_root_path()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5.67, 2.5))

    # Clustering
    sns.lineplot(
        cluster_stats, x='threshold', y='precision', ax=ax1, color='darkred',
        linewidth=2.8, alpha=0.6, label='Model')
    sns.lineplot(
        cluster_stats, x='threshold', y='recall', ax=ax2, color='darkred', linewidth=2.8, alpha=0.6)

    # Land filtering
    sns.lineplot(
        ocean_stats, x='threshold', y='precision', ax=ax1, color='indianred',
        alpha=0.6, linewidth=0.8, label='Object detection\nand land filtering')

    # All predictions
    sns.lineplot(
        all_stats, x='threshold', y='recall', ax=ax2, color='indianred',
        alpha=0.6, linewidth=0.8, linestyle='--')
    sns.lineplot(
        all_stats, x='threshold', y='precision', ax=ax1, color='indianred',
        alpha=0.6, linewidth=0.8, linestyle='--', label='Object detection')

    # Labels and style
    ax1.set(xlabel='Confidence threshold', ylabel='Precision')
    ax2.set(xlabel='Confidence threshold', ylabel='Recall')

    ax1.set_ylabel('Precision', fontname='Myriad Pro', fontsize=8)
    ax1.set_xlabel('Model confidence threshold', fontname='Myriad Pro', fontsize=8)
    stylize_axes(ax1)

    ax2.set_ylabel('Recall', fontname='Myriad Pro', fontsize=8)
    ax2.set_xlabel('Model confidence threshold', fontname='Myriad Pro', fontsize=8)
    stylize_axes(ax2)

    ax1.set_xticks([0., 0.2, 0.4, 0.6, 0.8, 1.], [0., 0.2, 0.4, 0.6, 0.8, 1.], fontname='Myriad Pro', fontsize=8)
    ax2.set_xticks([0., 0.2, 0.4, 0.6, 0.8, 1.], [0., 0.2, 0.4, 0.6, 0.8, 1.], fontname='Myriad Pro', fontsize=8)
    ax1.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.], [0., 0.2, 0.4, 0.6, 0.8, 1.], fontname='Myriad Pro', fontsize=8)
    ax2.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.], [0., 0.2, 0.4, 0.6, 0.8, 1.], fontname='Myriad Pro', fontsize=8)

    ax1.legend(frameon=False, fontsize=8, prop={'family': 'Myriad Pro', 'size': 8})

    plt.tight_layout()
    fig.savefig(f"{root_dir}/output/paper_figures/performance_across_confidence_pipeline_disagg.pdf",
                dpi=300, format='pdf',
                bbox_inches='tight')


def main():
    # Get root directory
    root_dir = file_utils.get_root_path()
    os.makedirs(f'{root_dir}/output/paper_figures', exist_ok=True)

    # Optimal Hyperparameters
    CONF_THRESH = 0.785
    DISTANCE_THRESHOLD = 50
    MIN_CLUSTER_SIZE = 5
    selected_map = map_year_to_image_pass_opt2

    datasets = load_datasets_for_model_evaluation(
        detections_path=f"{root_dir}/output/detections.geojson",
        landshape_path=f"{root_dir}/data/shapefiles/clean/france_final_land_filter/france_final_land_filter.shp",
        base_dir=str(root_dir))

    all_images = datasets['all_images']
    detections = datasets['detections']
    sampled_images = datasets['sampled_images']
    labels = datasets['labels']

    # Number of false positives in the raw detections
    sample_detections = detections[detections['image'].isin(sampled_images['image'])].copy()
    sample_detections['tp'] = get_tp(sample_detections, labels.to_crs(sample_detections.crs))

    print(f"Percentage of false positives in the raw detections: "
          f"{100 - sample_detections['tp'].sum() / len(sample_detections) * 100}")

    # Number of false positives in the ocean detections
    sample_detections_ocean = sample_detections[sample_detections['bucket'] != 'land'].copy()
    total_fp = sample_detections.loc[sample_detections['tp'] == False]
    ocean_fp = total_fp.loc[total_fp['bucket'] != 'land']
    print(f"Percentage of false positives that are dropped by using the land filter: "
          f"{100 - len(ocean_fp) / len(total_fp) * 100}")

    # Precision and recall curves
    assert sample_detections.crs == labels.crs
    assert detections.crs == labels.crs

    # All predictions
    bucket_info = get_bucket_info_table(all_images)
    all_stats = get_sample_stats_at_thresholds(
        labels, sample_detections, np.linspace(0, 1, 100))

    # Land filtering
    bucket_info_ocean = bucket_info[bucket_info.index != 'land'].copy()
    ocean_stats = get_sample_stats_at_thresholds(
        labels, sample_detections_ocean.to_crs(labels.crs),
        np.linspace(0, 1, 100))

    # Clustering
    cages = sample_detections_ocean.copy()
    cages.to_crs('EPSG:3035', inplace=True)

    sample_detections_cluster = DBSCAN_cluster(
        cages=cages, facilities_path='', distance_threshold=DISTANCE_THRESHOLD, amnt_min_clusters=MIN_CLUSTER_SIZE,
        include_area=False, save=False, return_detections=True, cluster_variable='year'
    )
    bucket_info_cluster = bucket_info_ocean
    cluster_stats = get_sample_stats_at_thresholds(
        labels=labels, sample_predictions=sample_detections_cluster.to_crs(labels.crs),
        thresholds=np.linspace(0, 1, 100))

    plot_new_figure(all_stats, ocean_stats, cluster_stats)

    # Stats at specific confidence thresholds
    stats = pd.concat(
        [all_stats, ocean_stats, cluster_stats], axis=0, keys=['all', 'ocean', 'ocean + clustered'],
        names=['domain']).droplevel(1).reset_index()

    print('A combination of land filtering and filtering predictions with a model score lower than..')
    print(stats.loc[stats['domain'] == 'all'].sort_values('precision').head(1))
    print(stats.loc[(stats['domain'] == 'ocean') & (stats['threshold'] >= 0.75)].head(1))

    # Stats: all
    print(stats.loc[(stats['domain'] == 'all') & (stats['threshold'] >= 0.5)].sort_values('precision').head(1))
    print(stats.loc[(stats['domain'] == 'all') & (stats['threshold'] >= 0.8)].sort_values('precision').head(1))

    #
    print(stats.loc[(stats['domain'] == 'ocean') & (stats['threshold'] >= 0.8)].sort_values('precision').head(1))
    print(stats.loc[(stats['domain'] == 'ocean + clustered') & (stats['threshold'] >= 0.8)].sort_values('precision').head(1))


if __name__ == '__main__':
    main()
