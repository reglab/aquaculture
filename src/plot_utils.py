import geopandas as gpd
import folium
from reglab_utils.geo.visualization import create_map


def plot_facility_map(facility_gdf: gpd.GeoDataFrame, name_var: str, grouping_var: str = None, save_file: str = None):
    facilities_map = facility_gdf.to_crs('EPSG:4326')

    center_x = facilities_map.centroid.x.mean()
    center_y = facilities_map.centroid.y.mean()

    # Generate map
    map = create_map(
        center_y,
        center_x,
    )

    if grouping_var:
        for group in facilities_map[grouping_var].unique():
            fg = folium.FeatureGroup(name=f'{group}')

            fac_group = facilities_map[facilities_map[grouping_var] == group]
            folium.GeoJson(
                fac_group.__geo_interface__,
                # style_function=lambda x: {'color': colors[x['properties']['pass']], 'alpha': 0.5},
                tooltip=folium.GeoJsonTooltip(fields=[name_var], labels=True)
            ).add_to(fg)

            fg.add_to(map)

        folium.LayerControl().add_to(map)
    else:
        fg = folium.FeatureGroup(name=f'Facilities')

        folium.GeoJson(
            facilities_map.__geo_interface__,
            # style_function=lambda x: {'color': colors[x['properties']['pass']], 'alpha': 0.5},
            tooltip=folium.GeoJsonTooltip(fields=[name_var], labels=True)
        ).add_to(fg)

        fg.add_to(map)

    if save_file:
        map.save(save_file)
    return map
