import pandas as pd
import numpy as np
from func import unique_pd, find_and_replace_not_num_values, isfloat
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
'''import data'''
path_part_A = r'Data/GWQ_2000-2009 coord corrected_csv.csv'
df_A = pd.read_csv(path_part_A, low_memory=False)
path_part_B = r'Data/GWQ_2010-2018 coord corrected_csv.csv'
df_B = pd.read_csv(path_part_B, low_memory=False)

'''mathup name columns'''
df_B.rename(columns={'LATITUD': 'LATITUDE', 'LOGITUD': 'LONGITUDE'}, inplace=True)
df = pd.concat([df_A, df_B], axis=0)

df.reset_index(inplace=True)


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


'''preper df['RSC']'''
df['RSC'].replace([' ', '#REF!', '#VALUE!', 'ND'], None, inplace=True)


'''this part we fix the numrical columns'''
columns = ['PH', 'EC', 'TH',
       'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE',
       'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE', 'SAR',
        'SiO2', 'TDS', 'Turbidity', '%Na', 'Arsenic', 'RSC']

'''drop the columns'''



for col in columns:
    print(col)
    df[col].replace(['-', '<5', '<1', 'B', '*'], None, inplace=True)

    df[col], _ = find_and_replace_not_num_values(df[col], replace_to=None, inplace=True, astype=True, lops=True, list_values=True)


'''df ready to fill null '''
# df_r = df.copy()
# df_r.to_csv('Data/df_.csv', index=False)

'''now we can look statistical numeric columns'''
df_numeric = df.loc[:, 'PH':'Arsenic']

describe = df_numeric.describe()
corr = df_numeric.corr()

'''this part make histplot for all column'''
# cols = ['PH', 'EC', 'TH',
#        'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE',
#        'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDcuont(nic']
# for col in tqdm(cols):
#     sns.histplot(df[col])
#     plt.show()

# info = df.count()
# info.to_excel("info.xlsx")

print('end')



'''we can look thet amaunt of graoundwater growduring in time '''
unique_pd(df.groupby('WRIS ID').count()['year'][1:])

'''we sink to add new feature: fluoride t-1, so hear we can swo that no '''
df_id_per_year = df.loc[:, ['WRIS ID', 'year', 'FLUORIDE']]
df_id_per_year_sort = df_id_per_year.sort_values(['year', 'WRIS ID'])
df_id_per_year_std = df_id_per_year.groupby(['WRIS ID']).std()