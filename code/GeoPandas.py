import pandas as pd
import numpy as np
from func import unique_pd, find_and_replace_not_num_values, isfloat
# import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import geopandas
import rasterio
from rasterio.plot import show
from shapely.geometry import Point

path_data = r'df_r.csv'
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
    df[col].replace(['-', '<5', '<1', 'B', '*'], None, inplace=True)

    df[col], _ = find_and_replace_not_num_values(df[col], replace_to=0, inplace=True, astype=True, lops=True, list_values=True)


gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.LONGITUDE, df.LATITUDE))

coord_list = [(x,y) for x,y in zip(gdf['geometry'].x, gdf['geometry'].y)]
coord_list

gdf['value'] = [x[0] for x in dataset.sample(coord_list)]
gdf.head()