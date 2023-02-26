import pandas as pd
from func import *
import geopandas
import rasterio
from rasterio.plot import show
from shapely.geometry import Point

from os import listdir
from os.path import isfile, join

path_data = r'Data/df_eda.csv'
df = pd.read_csv(path_data, low_memory=False)
gdf = geopandas.GeoDataFrame(df)
df_fill_TH = pd.read_csv(r'Data/df_fill_TH.csv', low_memory=False)


def load_rester(path_raster, data):
    coord_list = None
    if coord_list is None:
        coord_list = [(x, y) for x, y in zip(data['geometry'].x, data['geometry'].y)]

    raster_list = [f for f in listdir(path_raster) if isfile(join(path_raster, f))]
    for i in raster_list:
        print(i)
        raster_df = rasterio.open(f'{path_raster}/{i}')
        raster_df = [x[0] for x in raster_df.sample(coord_list)]
        gdf[i] = raster_df

    return gdf



columns_coordinates = ['LATITUDE', 'LONGITUDE']

for col in columns_coordinates:
    gdf[col], _ = find_and_replace_not_num_values(gdf[col], replace_to=0, inplace=True, astype=True, lops=True, list_values=True)

gdf = geopandas.GeoDataFrame(gdf, geometry=geopandas.points_from_xy(gdf.LONGITUDE, gdf.LATITUDE))


path_raster_list = ['New India Maps/General Raster', 'New India Maps/Ardity', 'New India Maps/precipitation']
for path in path_raster_list:
    gdf = load_rester(path, gdf)
# gdf = pd.DataFrame([load_rester(path, gdf) for path in path_raster_list])

feature_raster_list = ['Ardity', 'precipitaion']
for raster in feature_raster_list:
    for year in range(2010, 2019):
        print(f'{raster} {year}.tif')
        gdf.loc[gdf['year'] != year, f'{raster} {year}.tif'] = None

df_marge = pd.concat([gdf.drop(df_fill_TH.columns.to_list(), axis=1), df_fill_TH], axis=1)
df_marge.copy().to_csv('Data/gdf.csv', index=False)

import json

def save_dict_to_file(dict_obj, file_path):
    """
    Saves a dictionary to a file as a JSON string.

    Args:
        dict_obj (dict): The dictionary to save.
        file_path (str): The path to the file where the dictionary should be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(dict_obj, file)

country_list_of_dakan = ['Andhra Pradesh', 'Dadra And Nagar Haveli', 'Goa', 'Karnataka',
                         'Kerala', 'Maharashtra', 'Odisha', 'Pondicherry', 'Tamil Nadu', 'Telangana', ]
country_list_of_himalayan = ['Arunachal Pradesh', 'Assam', 'Himachal Pradesh', 'Jammu And Kashmir',
                             'Meghalaya', 'Nagaland', 'Tripura', 'Uttarakhand', ]
country_list_of_lowland = ['Bihar', 'Chhattisgarh', 'Delhi', 'Gujarat', 'Haryana', 'Jharkhand',
                           'Punjab', 'Rajasthan', 'Uttar Pradesh', 'West Bengal']
dict_country_list = {'dakan': country_list_of_dakan, 'himalayan': country_list_of_himalayan,
                   'lowland': country_list_of_lowland}

save_dict_to_file(dict_country_list, 'Data/dict_country_list.json')


print('end')

