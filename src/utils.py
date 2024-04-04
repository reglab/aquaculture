import os
from google.cloud import storage
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio.features
from shapely.geometry import GeometryCollection, box, polygon
from shapely.affinity import affine_transform
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import src.file_utils as file_utils

LARGE_TIF_SIZE = 1024*6
IM_WIDTH = 1024
IM_HEIGHT = 1024
CRS_DICT = {'mapping': 3857, 'area': 3035}

tqdm.pandas()


def load_download_bboxes(
        path: str = f"{file_utils.get_root_path()}/data/wanted_bboxes.csv"
) -> gpd.GeoDataFrame:
    """Load the download bounding boxes.

    Args:
        path: The path to the download bounding boxes.

    Returns:
        A GeoDataFrame containing the download bounding boxes.
    """

    download_bboxes = pd.read_csv(path, index_col=0)

    return gpd.GeoDataFrame(
        download_bboxes,
        geometry=gpd.GeoSeries.from_wkt(download_bboxes['geometry']),
        crs="EPSG:3857"
    )


def mark_land_images(
    images: gpd.GeoDataFrame,
    land_gdf: gpd.GeoDataFrame,
    land_indent: float = 5,
    projected_crs: str = "EPSG:3857"
) -> gpd.GeoSeries:
    """Returns a GeoSeries denoting whether an image only contains land.
    
    Args:
        images: A gdf of images.
        land_gdf: a gdf representing land mass.
        land_indent: amount to indent land geometry in projected CRS units.
        projected_crs: CRS to calculate land indent in.
        
    Returns:
        A series where true means an image only contains land.
    """
    buffered_land = gpd.GeoDataFrame(
        geometry=land_gdf.to_crs(projected_crs).buffer(-land_indent)
    ).dissolve()
    
    return images.index.isin(
        images.sjoin(buffered_land.to_crs(images.crs), how='inner', predicate='within').index.unique()
    )


def load_cf_labels() -> gpd.GeoDataFrame:
    """Load the CloudFactory (human-annotated) labels for the French Mediterranean.
    Returns:
        A GeoDataFrame containing the CloudFactory labels.
    """

    cf_labels = gpd.read_file(f'{file_utils.get_root_path()}/output/humanlabels.geojson')

    # Ensure CRS is EPSG:3857
    cf_labels.to_crs(f"EPSG:{CRS_DICT['mapping']}", inplace=True)

    return cf_labels


def load_cf_images() -> pd.DataFrame:
    """Load a DataFrame of the images submitted to CloudFactory.

    Returns:
        A DataFrame containing the CloudFactory images.
    """
    cf_images = pd.read_csv(f'{file_utils.get_root_path()}/output/cf_images.csv')
    return cf_images

  
def load_Trujillo_locations_deduped(
        path: str = f"{file_utils.get_root_path()}/data/aquaculture_med_dedupe.csv") -> gpd.GeoDataFrame:
    """Loads a GeoDataFrame of Trujillo. et al (2012) deduplicated aquaculture locations.
    
    Args:
        path: path to csv of deduped locations.
        
    Returns:
        A gdf of aquaculture locations.
    """
    Trujillo_locations = pd.read_csv(path)
    Trujillo_locations = gpd.GeoDataFrame(
        Trujillo_locations,
        geometry=gpd.GeoSeries.from_xy(Trujillo_locations['lon'], Trujillo_locations['lat']),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    return Trujillo_locations


def map_year_to_image_pass_opt2(year) -> str:
    if 2000 <= year <= 2004:
        return '2000-2004'
    elif 2005 <= year <= 2009:
        return '2005-2009'
    elif 2010 <= year <= 2012:
        return '2010-2012'
    elif 2013 <= year <= 2015:
        return '2013-2015'
    elif 2016 <= year <= 2018:
        return '2016-2018'
    elif 2019 <= year <= 2021:
        return '2019-2021'
    else:
        return 'No group'


def stylize_axes(ax):
    """Removes top and right spines from a matplotlib axis.

    Args:
        ax: a matplotlib axis.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def get_cage_min_and_max_areas(cages: gpd.GeoDataFrame, area_crs: str = f"EPSG:{CRS_DICT['area']}") -> pd.Series:
    """Gets the minimum and maximum area of cages.

    Args:
        cages: a GeoDataFrame of cages.

    Returns:
        The maximum area of a cage in the label file.
    """

    return cages.to_crs(area_crs).groupby("type").apply(lambda x: x.area.max())


def get_french_image_data(
        download_bboxes: gpd.GeoDataFrame,
        image_file: str) -> tuple[box, polygon.Polygon, int, int, int, int]:
    """
    Returns the box, the large TIF box that the image belongs to, and the year of an image according
    to the API (in EPSG:3857)
    :param download_bboxes: gdf of large boxes used to download the French imagery
    :param image_file: image path in the GCS bucket
    :param large_tif_size:
    :return:
    """
    # Get offsets, year and large bbox
    name, bbox_ind, x_offset, y_offset = os.path.basename(image_file).replace('.jpeg', '').split('_')
    year = name[-4:]
    large_tif_bbox = download_bboxes.loc[int(bbox_ind), 'geometry']
    xmin_m, ymin_m, xmax_m, ymax_m = large_tif_bbox.bounds

    # Top-left corner
    x_loc = 0 + int(x_offset)
    y_loc = 0 + int(y_offset)

    x_min = x_loc * ((xmax_m - xmin_m) / LARGE_TIF_SIZE) + xmin_m  # m per pixel -- left side = lower x
    y_min = ymax_m - y_loc * ((ymax_m - ymin_m) / LARGE_TIF_SIZE)  # m per pixel -- bottom = lower y

    # Bottom-right corner
    x_loc = IM_WIDTH + int(x_offset)
    y_loc = IM_HEIGHT + int(y_offset)

    x_max = x_loc * ((xmax_m - xmin_m) / LARGE_TIF_SIZE) + xmin_m
    y_max = ymax_m - y_loc * ((ymax_m - ymin_m) / LARGE_TIF_SIZE)

    # Create box
    img_box = box(minx=x_min, miny=y_min, maxx=x_max, maxy=y_max)
    return img_box, large_tif_bbox, year, bbox_ind, x_offset, y_offset


def get_french_image_boxes(
        download_bboxes: gpd.GeoDataFrame,
        path_to_image_gdf: str,
        gcs_bucket_name: str = 'image-hub',
        user_project_name: str = 'law-cafo',
        gcs_im_path: str = 'coastal_french_data/jpegs'
) -> gpd.GeoDataFrame:
    """
    Obtains the image boxes and image years by looping over the imagery in GCP. These
    are the raw image boxes, without accounting for any kind of de-duplication.
    :param download_bboxes: gdf of large boxes used to download the French imagery
    :param path_to_image_gdf: location to save/read the gdf produced by this function
    :param gcs_bucket_name: name of the bucket on GCS where the French data is stored
    :param user_project_name: name of the project on GCS
    :param gcs_im_path: path to the jpeg files within the GCS bucket
    :return:
    """
    if not os.path.exists(path_to_image_gdf):
        print('[INFO] Loading image data from GCP')

        # Get image names
        storage_client = storage.Client()
        bucket = storage.Bucket(storage_client, gcs_bucket_name, user_project=user_project_name)
        all_blobs = list(storage_client.list_blobs(bucket, prefix=gcs_im_path))
        image_files = [b.name for b in all_blobs]

        # Create GDF of image boxes
        image_boxes = pd.DataFrame()
        for img_file in tqdm(image_files):
            if '(' in img_file:
                continue

            img_box, _, year, bbox_ind, x_offset, y_offset = get_french_image_data(download_bboxes=download_bboxes,
                                                                                   image_file=img_file)
            img_dict = {
                'year': [year], 'geometry': [img_box], 'bbox_ind': [bbox_ind],
                'x_offset': [x_offset], 'y_offset': [y_offset]
            }
            image_boxes = pd.concat([image_boxes, pd.DataFrame.from_dict(img_dict)])

        image_boxes = gpd.GeoDataFrame(image_boxes, crs=download_bboxes.crs)
        image_boxes.to_file(path_to_image_gdf, driver='GeoJSON')

    else:
        print('[INFO] Loading from file...')
        image_boxes = gpd.read_file(path_to_image_gdf)
    return image_boxes


def deduplicate_download_boxes(bboxes: gpd.GeoDataFrame, path: str) -> gpd.GeoDataFrame:
    """
    Deduplicates wanted_bboxes to generate new polygons for each bbox index.
    :param bboxes:
    :param path: location where deduped data is stored if it already exists. If not, it saves
    to this location.
    :return:
    """
    if os.path.exists(path):
        dedup_boxes = gpd.read_file(path)
        return dedup_boxes

    bboxes.to_crs("EPSG:3857", inplace=True)
    bboxes['bbox_ind'] = bboxes.index

    dedup_boxes = bboxes.iloc[[0]]

    for bbox_ind in tqdm(range(1, bboxes.shape[0])):
        # Get geometry
        bbox_row = bboxes.iloc[[bbox_ind]]

        # Get existing coverage
        existing_coverage = dedup_boxes.dissolve()

        new_row = gpd.overlay(bbox_row, existing_coverage, how='difference')

        # Check if overlap
        dedup_boxes = pd.concat([dedup_boxes, new_row])
        dedup_boxes = gpd.GeoDataFrame(dedup_boxes, crs=bboxes.crs)

    # Save
    dedup_boxes.to_file(path, driver='GeoJSON')
    return dedup_boxes


def deduplicate_gdf_with_bboxes(dedup_boxes, gdf_todedup, path: str = None):
    """
    Uses the de-duplicated bounding boxes to de-duplicate any geo dataframe.
    :param dedup_boxes:
    :param gdf_todedup:
    :param path: location to save or read the file if this function has already been run
    on a specific dataset before.
    :return:
    """
    if path:
        print('[INFO] Loading deduplicated gdf from path...')
        if os.path.exists(path):
            return gpd.read_file(path)

    gdf_crs = gdf_todedup.crs
    dedup_boxes.to_crs("EPSG:3857", inplace=True)
    gdf_todedup.to_crs("EPSG:3857", inplace=True)

    if 'bbox_ind' not in gdf_todedup.columns:
        raise Exception('[Error] gdf should include bbox_ind column')
    gdf_todedup['bbox_ind'] = gdf_todedup['bbox_ind'].astype(int)
    tqdm.pandas()

    # Drop all the observations from gdf that were in bboxes that got fully dropped
    gdf_todedup = gdf_todedup.loc[gdf_todedup['bbox_ind'].isin(dedup_boxes['bbox_ind'].unique())].copy()

    # Drop observations that don't fall in the new geometries for the bounding boxes
    def intersecting_geom(row, gdf):
        box_geom = gdf.loc[gdf['bbox_ind'] == row['bbox_ind']].iloc[0]['geometry']
        obs_geom = row['geometry']
        return obs_geom.intersection(box_geom)

    gdf_todedup['geometry'] = gdf_todedup.progress_apply(
        lambda row: intersecting_geom(row=row, gdf=dedup_boxes), axis=1)

    # Drop geometries that become empty
    gdf_todedup['empty'] = gdf_todedup['geometry'].apply(lambda g: g.is_empty)
    gdf_todedup = gdf_todedup.loc[~gdf_todedup['empty']]
    gdf_todedup.drop('empty', axis=1, inplace=True)

    # Convert back to original CRS
    gdf_todedup.to_crs(gdf_crs, inplace=True)

    if path:
        gdf_todedup.to_file(path, driver='GeoJSON')

    return gdf_todedup


def is_blank(img_bytes=None, im=None):
    '''
    check if downloaded image bytes are actually a blank image
    where "blank" means either all white or all black

    Input:
        img_bytes: bytes of image downloaded from wms

    Returns: boolean for whether or not the image is blank
    '''
    if not im and img_bytes:
        im = Image.open(BytesIO(img_bytes))

    extrema = im.convert("L").getextrema()
    if extrema == (0, 0):
        return True
    elif extrema == (1, 1):
        return True
    elif extrema == (255, 255):
        # all white
        return True
    elif extrema[0] >= 250. and extrema[1] >= 250.:
        return True
    else:
        return False


def is_partly_blank(im: Image) -> bool:
    """
    Determines whether an image has fully blank spaces at the row or column level
    :param im:
    :return:
    """
    row_avg = np.average(im, axis=(1, 2))
    num_blank_rows = np.where(row_avg >= 250.)
    num_blank_rows = num_blank_rows[0].shape[0]

    col_avg = np.average(im, axis=(0, 2))
    num_blank_cols = np.where(col_avg >= 250.)
    num_blank_cols = num_blank_cols[0].shape[0]

    if num_blank_cols + num_blank_rows > 0:
        return True

    return False


def generate_image_file_name_str(d: dict, extension='jpeg') -> str:
    """
    Returns the image file name based on the year, bbox index, x offset and y offset.
    :param d: a dictionary containing these items
    :param extension: the file extension
    :return:
    """
    name = f"{d['year']}_{d['bbox_ind']}_{d['x_offset']}_{d['y_offset']}"
    image_file = f'ORTHOIMAGERY.ORTHOPHOTOS{name}.{extension}'
    if int(d['year']) == 2021:
        image_file = f'ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.{name}.{extension}'
    return image_file


def generate_image_specs_from_file_name(file: str) -> dict:
    name, bbox_ind, x_offset, y_offset = os.path.basename(file).replace('.jpeg', '').split('_')
    year = name[-4:]
    return {'name': name, 'bbox_ind': bbox_ind, 'x_offset': x_offset, 'y_offset': y_offset, 'year': year}


def remove_white_image_boxes(
        img_boxes: gpd.GeoDataFrame,
        path_to_clean_image_gdf: str,
        gcs_bucket_name: str = 'image-hub',
        gcs_im_path: str = 'coastal_french_data/jpegs'
) -> gpd.GeoDataFrame:
    """
    Removes image boxes with blank images and modifies the geometry of image boxes that have partly
    blank imagery.
    :param img_boxes: gpd of the image boxes for all years
    :param path_to_clean_image_gdf: file location to save final output
    :param gcs_bucket_name: name of the GCS bucket where French data is stored
    :param gcs_im_path: path to the jpeg files within the GCS bucket
    :return: gpd of image boxes with geometries containing non-blank imagery
    """
    if not os.path.exists(path_to_clean_image_gdf):
        print('[INFO] Finding blank images..')

        if not os.path.exists(f'{file_utils.get_root_path()}/data/image_boxes_blank_key.csv'):
            print('[INFO] Checking all images for whitespace')
            img_boxes_dd = dd.from_pandas(
                img_boxes[['year', 'bbox_ind', 'x_offset', 'y_offset']], npartitions=5)

            gcs_client = storage.Client()
            gcs_bucket = gcs_client.get_bucket(gcs_bucket_name)

            def get_image_blank_status(row):
                # Get file name
                image_file = generate_image_file_name_str(d=row, extension='jpeg')

                # Check image
                blob = gcs_bucket.blob(os.path.join(gcs_im_path, image_file))
                img = Image.open(BytesIO(blob.download_as_bytes()))

                if is_blank(im=img):
                    return 'blank'
                elif is_partly_blank(im=img):
                    return 'partly blank'
                else:
                    return 'complete'

            with ProgressBar():
                result = img_boxes_dd.apply(get_image_blank_status, axis=1, meta=('image_status', 'str'))
                img_boxes['image_status'] = result.compute()
            print('Saving blank key data frame')

            img_boxes.drop('geometry', axis=1, inplace=True)
            img_boxes.to_csv(f'{file_utils.get_root_path()}/data/image_boxes_blank_key.csv')

        # Convert to gdf and save final version
        blank_key_df = pd.read_csv(f'{file_utils.get_root_path()}/data/image_boxes_blank_key.csv')

        # Drop blank images
        blank_key_df = blank_key_df.loc[blank_key_df['image_status'] != 'blank']

        # Re-create image file name
        blank_key_df['image_file'] = blank_key_df.apply(
            lambda row: generate_image_file_name_str(d=row, extension='jpeg'), axis=1)
        img_boxes['image_file'] = img_boxes.apply(
            lambda row: generate_image_file_name_str(d=row, extension='jpeg'), axis=1)

        # Partially blank images: modify geometry
        pb_key_df = blank_key_df.loc[blank_key_df['image_status'] == 'partly blank'].copy()
        final_df = img_boxes.loc[img_boxes['image_file'].isin(pb_key_df['image_file'].unique())].copy()

        print('[INFO] Correcting geometry for partly blank images...')
        gcs_client = storage.Client()
        gcs_bucket = gcs_client.get_bucket(gcs_bucket_name)
        final_df['geometry'] = final_df.progress_apply(
            lambda row: correct_partly_blank_geom(image_dict=row, bucket=gcs_bucket), axis=1)

        # Drop images initially marked as partly blanked but that are actually blank
        final_df['empty'] = final_df['geometry'].apply(lambda g: g.is_empty)
        final_df = final_df.loc[~final_df['empty']]
        final_df.drop('empty', axis=1, inplace=True)

        # Append only the complete imagery
        complete_key_df = blank_key_df.loc[blank_key_df['image_status'] == 'complete'].copy()
        final_df = pd.concat([
            final_df,
            img_boxes.loc[img_boxes['image_file'].isin(complete_key_df['image_file'].unique())].copy()])
        final_df = gpd.GeoDataFrame(final_df, crs=img_boxes.crs)
        print(f'[INFO] Saving final output. {len(final_df)} final geometries')
        final_df.to_file(path_to_clean_image_gdf, driver='GeoJSON')

    print('[INFO] Loading from file..')
    clean_image_boxes = gpd.read_file(path_to_clean_image_gdf)
    return clean_image_boxes


def correct_partly_blank_geom(
        image_dict: dict,
        bucket: storage.bucket.Bucket,
        gcs_im_path: str = 'coastal_french_data/jpegs'
) -> polygon.Polygon:
    """
    Returns a modified polygon for an image that reflects only the non-blank areas in the image. If the
    image is fully blank, it returns an empty polygon.
    :param image_dict: dictionary including keys necessary to re-generate the image name as stored in GCP
    :param bucket: gcp bucket where image is stored
    :param gcs_im_path: path to the jpeg files within the GCS bucket
    :return: modified polygon
    """

    # Get file name
    image_file = generate_image_file_name_str(d=image_dict, extension='jpeg')

    # Get image
    blob = bucket.blob(os.path.join(gcs_im_path, image_file))
    img = Image.open(BytesIO(blob.download_as_bytes()))

    # Modify geometry
    complete_geom = image_dict['geometry']

    # First: create a boolean mask of non-blank parts of the image
    img_np = np.asarray(img)
    non_blank = np.max(img_np, axis=2)
    non_blank = (non_blank < 250.)
    non_blank = non_blank.astype(np.int32)

    # Get shapes -> get the largest non-blank geom
    shapes = rasterio.features.shapes(non_blank, connectivity=8)
    polygons = [polygon.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]
    max_poly = None
    max_poly_area = 0
    for poly in polygons:
        if poly.area > max_poly_area:
            max_poly = poly
            max_poly_area = poly.area
    if max_poly is None:
        return polygon.Polygon()

    # Transform and simplify
    img_bbox = box(*complete_geom.bounds)
    t = rasterio.transform.from_bounds(*img_bbox.bounds, width=IM_WIDTH, height=IM_HEIGHT)
    max_poly = affine_transform(max_poly, [t.a, t.b, t.d, t.e, t.xoff, t.yoff])
    max_poly = max_poly.simplify(0.5)

    return max_poly


def load_final_image_boxes(main_dir: str) -> gpd.GeoDataFrame:
    """
    Generates the gdf of image boxes, accounting for: 1) spatial deduplication from the large bounding boxes; and
    2) whitespace in the imagery.
    :param main_dir:
    :return: gdf of image boxes with modified geometries.
    """
    if os.path.exists(os.path.join(main_dir, 'data/image_boxes_years_rmblank.geojson')):
        print('[INFO] Loading image boxes (without whitespace) from file')
        rmblank_image_boxes = remove_white_image_boxes(
            img_boxes=None, path_to_clean_image_gdf=os.path.join(main_dir, 'data/image_boxes_years_rmblank.geojson'))
    else:
        print('[INFO] Generating image boxes (without whitespace)')
        # Get large boxes and de-duplicate these
        download_bboxes = load_download_bboxes(os.path.join(main_dir, "data/wanted_bboxes.csv"))
        dedup_boxes = deduplicate_download_boxes(
            download_bboxes, path=os.path.join(main_dir, "data/wanted_bboxes_dedup.csv"))

        # Within-period deduplication: obtain the image boxes and spatially deduplicate
        image_boxes = get_french_image_boxes(
            download_bboxes=download_bboxes,
            path_to_image_gdf=os.path.join(main_dir, 'data/image_boxes_years.geojson'))
        image_boxes_dedup = deduplicate_gdf_with_bboxes(
            dedup_boxes=dedup_boxes, gdf_todedup=image_boxes,
            path=os.path.join(main_dir, 'data/image_boxes_years_dedup.geojson'))

        # Locate whitespace in imagery
        rmblank_image_boxes = remove_white_image_boxes(
            img_boxes=image_boxes_dedup,
            path_to_clean_image_gdf=os.path.join(main_dir, 'data/image_boxes_years_rmblank.geojson'))
    return rmblank_image_boxes

