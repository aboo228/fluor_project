import pandas as pd
import numpy as np
from func import unique_pd, find_and_replace_not_num_values, isfloat
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


path_part_A = r'GWQ_2000-2009 coord corrected_csv.csv'
df_A = pd.read_csv(path_part_A, low_memory=False)
path_part_B = r'GWQ_2010-2018 coord corrected_csv.csv'
df_B = pd.read_csv(path_part_B, low_memory=False)


df_B.rename(columns={'LATITUD': 'LATITUDE', 'LOGITUD': 'LONGITUDE'}, inplace=True)
df = pd.concat([df_A, df_B], axis=0)

df.reset_index(inplace=True)

# df.loc[:,'PROJECT_NAME':]
# df.columns


'''this part show that most the sit is had less then 5 instance'''
# unique = unique_pd(df['SITE_ID'], df['SITE_ID'])
# unique = unique.drop('New')
# sns.histplot(unique)
# plt.show()

'''this part make histplot for all column'''
# cols = ['SITE_ID', 'WRIS ID', 'LATITUDE', 'LONGITUDE', 'SITE_TYPE',
#        'STATE_NAME', 'DISTRICT_NAME', 'TAHSIL_NAME', 'BLOCK_NAME', 'SITE_NAME',
#        'BASIN_NAME', 'PROJECT_NAME', 'year']
# for col in tqdm(cols):
#     unique = unique_pd(df[col], df[col])
#     sns.histplot(unique)
#     plt.show()


# df['TOT_ALKALINITY'], list_n = find_and_replace_not_num_values(df['TOT_ALKALINITY'], inplace=True, astype=True, lops=True, list_values=True)




'''this part we fix the numrical columns'''
columns = ['PH', 'EC', 'TH',
       'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE',
       'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE', 'SAR',
       'RSC', 'SiO2', 'PO4', 'TDS', 'Turbidity', '%Na', 'Arsenic']


for col in columns:
    print(col)
    df[col].replace(['-', '<5', '<1', 'B', '*'], None, inplace=True)
    df[col], _ = find_and_replace_not_num_values(df[col], replace_to=None, inplace=True, astype=True, lops=True, list_values=True)

col_to_fix = ['LATITUDE', 'LONGITUDE', 'LR. No', 'RSC']

describe = df.loc[:, 'PH':'Arsenic'].describe()

# info = df.count()
# info.to_excel("info.xlsx")