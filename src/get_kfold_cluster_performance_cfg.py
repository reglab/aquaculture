import numpy as np

# path to file containing geocoded detections
detections_path = 'output/detections.geojson'

# Number of workers for multiprocessing
workers = 5

# Chunksize for multiprocessing
chunksize = 1

# Land shapefile path
landshape_path = "data/shapefiles/clean/france_final_land_filter/france_final_land_filter.shp"

# Number of folds
num_folds = 5

# Confidence thresholds
confidence_thresholds = np.arange(0.6, 1.01, 0.005)

# Distance thresholds
distance_thresholds = np.arange(10, 151, 20)

# Minimum cluster sizes to test
minimum_cluster_sizes = np.arange(1, 11, 1, dtype=int)

# Random state
random_state = 1

# Output path
output_path = f"output/cross_validation/{num_folds}_fold_results_{random_state}.csv"

