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

path_data = r'Data/df_eda.csv'
data = pd.read_csv(path_data, low_memory=False)
path_raster = r'New India Maps/sand_content_1.5m.tif'
dataset = rasterio.open(path_raster)
raster = dataset.read()
df = geopandas.GeoDataFrame(data)
# df = geopandas.read_file(data)
gdf = geopandas.read_file(path_data)
# Importing fiona resulted in: DLL load failed while importing ogrext: The specified module could not be found.
# Importing pyogrio resulted in: No module named 'pyogrio'




columns = ['LATITUDE', 'LONGITUDE']
for col in columns:
    print(col)
    # df[col].replace(['-', '<5', '<1', 'B', '*'], None, inplace=True)

    df[col], _ = find_and_replace_not_num_values(df[col], replace_to=0, inplace=True, astype=True, lops=True, list_values=True)


gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.LONGITUDE, df.LATITUDE))

def load_rester(path_raster):
    coord_list = None
    if coord_list is None:
        coord_list = [(x, y) for x, y in zip(gdf['geometry'].x, gdf['geometry'].y)]

    print(path_raster)
    raster_df = rasterio.open(path_raster)
    raster_df = [x[0] for x in raster_df.sample(coord_list)]
    return raster_df


raster_list = ['PET', 'clay_content_1.5m', 'calcisols', 'ardity', 'alpha', 'AET', 'tri', 'slopes', 'silt_content_1.5m',
               'sand_content_1.5m', 'precipitation std', 'precipitation mean', 'pH_content_1.5m']


for i in tqdm(raster_list):
    path_raster = f'New India Maps/{i}.tif'
    gdf[i] = load_rester(path_raster)




#
# for i in tqdm(raster_list):
#
#
#     print(i)
#     path_raster = f'New India Maps/{i}.tif'
#     dataset = rasterio.open(path_raster)
#     raster = dataset.read()
#     gdf[i] = [x[0] for x in dataset.sample(coord_list)]

df_fill_TH = pd.read_csv(r'Data/df_fill_TH.csv', low_memory=False)

df_marge = pd.concat([gdf.drop(df_fill_TH.columns.to_list(), axis=1), df_fill_TH], axis=1)


df_marge = df_marge.copy()
df_marge.to_csv('Data/gdf.csv', index=False)
print('end')