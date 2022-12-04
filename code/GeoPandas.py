import pandas as pd
import numpy as np
from func import unique_pd, find_and_replace_not_num_values, isfloat
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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


# path_general_raster = 'New India Maps/General Raster'
# path_ardity = 'New India Maps/Ardity'
# path_precipitation = 'New India Maps/precipitation'
# path_list = [path_general_raster, path_ardity, path_precipitation]
columns_coordinates = ['LATITUDE', 'LONGITUDE']

for col in columns_coordinates:
    gdf[col], _ = find_and_replace_not_num_values(gdf[col], replace_to=0, inplace=True, astype=True, lops=True, list_values=True)

gdf = geopandas.GeoDataFrame(gdf, geometry=geopandas.points_from_xy(gdf.LONGITUDE, gdf.LATITUDE))
# gdf = pd.DataFrame([load_rester(path, gdf) for path in path_list])

path_raster_list = ['New India Maps/General Raster', 'New India Maps/Ardity', 'New India Maps/precipitation']
for path in path_raster_list:
    gdf = load_rester(path, gdf)

# gdf = pd.DataFrame(load_rester(path, gdf) for path in path_raster_list)
# general_raster = load_rester('New India Maps/General Raster', gdf)
# ardity = load_rester('New India Maps/Ardity', gdf)
# precipitation = load_rester('New India Maps/precipitation', gdf)
# gdf = pd.concat([general_raster.loc[:, 'AET.tif':], ardity.loc[:, 'Ardity 2010.tif':], precipitation.loc[:, 'precipitaion 2010.tif':]], axis=1)
#
df_marge = pd.concat([gdf.drop(df_fill_TH.columns.to_list(), axis=1), df_fill_TH], axis=1)
df_marge.copy().to_csv('Data/gdf.csv', index=False)

feature_raster_list = ['Ardity', 'precipitaion']
for raster in feature_raster_list:
    for year in range(2010, 2019):
        print(f'{raster} {year}.tif')
        gdf.loc[gdf['year'] != year, f'{raster} {year}.tif'] = None

df_marge = pd.concat([gdf.drop(df_fill_TH.columns.to_list(), axis=1), df_fill_TH], axis=1)
df_marge.copy().to_csv('Data/gdf.csv', index=False)
print('end')





