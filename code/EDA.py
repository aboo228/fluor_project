import pandas as pd
import numpy as np
from func import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

'''import data'''
path_part_A = r'Data/GWQ_2000-2009 coord corrected_csv.csv'
df_A = pd.read_csv(path_part_A, low_memory=False)
path_part_B = r'Data/GWQ_2010-2018 coord corrected_csv.csv'
df_B = pd.read_csv(path_part_B, low_memory=False)

'''math up name columns'''
df_B.rename(columns={'LATITUD': 'LATITUDE', 'LOGITUD': 'LONGITUDE'}, inplace=True)
df = pd.concat([df_A, df_B], axis=0)

df.reset_index(inplace=True)

df['RSC'].replace([' ', '#REF!', '#VALUE!', 'ND'], None, inplace=True)
df['STATE_NAME'][df['STATE_NAME'] == 'Uttarakhand '].replace('Uttarakhand', inplace=True)
df['SITE_TYPE'].replace(['DW', 'D.W.', 'Dug well', ' Dug Well', 'dug', 'Dugwell', 'DUGWWELL', 'Dug', 'DUG',
                         'WELL', 'D/W', 'Dug Welll', 'Dug ', 'DUG WELL', 'DCB', 'Mark II ', 'CGWB SPZ'],
                        'Dug Well', inplace=True)
df['SITE_TYPE'].replace(['MONITORING'], 'Monitoring', inplace=True)
df['SITE_TYPE'].replace(['BOREWELL', 'BW', 'BW '], 'Bore Well', inplace=True)
df['SITE_TYPE'].replace(['TW', 'T/W', 'CYL. TW', 'T.W', ' TW'], 'Tube Well', inplace=True)
df['SITE_TYPE'].replace(['H/P', 'H.P', 'HP', 'Hand pump'], 'Hand Pump', inplace=True)
df['SITE_TYPE'].replace(['Mark II', 'markII', 'markII', 'Mark II '], 'Mark-II', inplace=True)
df['SITE_TYPE'].replace(['Cylindrical '], 'Cylindrical', inplace=True)
df['SITE_TYPE'].replace(['PZ'], 'Pz', inplace=True)
df['SITE_TYPE'].replace(['STW'], 'Sub T.W.', inplace=True)
df['SITE_TYPE'].replace(['TW M-II', 'Tw Mark-II', 'Mark-II T.W.', 'TW Mark-II', 'T.W.', 'Tw', 'markII TW',
                         'TW MK-II', 'Twmark-II''Mrk II', 'Stw', 'Submersible TW', 'TW ', 'Mark-II TW',
                         'TW mark II', 'T.W. Mark-II', 'MARK-II', 'Mark-II Tw', 'Mark-Il', 'SEW', 'Sub T.W',
                         'TIRUVALLUR'
                            , 'Twmark-II', 'Mrk II'], 'TW Mark II', inplace=True)

df['STATE_NAME'].replace(['Uttarakhand '], 'Uttarakhand', inplace=True)
df['STATE_NAME'].replace(['Daman & Diu'], 'Dadra And Nagar Haveli', inplace=True)
df['STATE_NAME'].replace(['Chandigarh'], 'Punjab', inplace=True)

'''we choose drop the columns that precent the null high from 70% '''
df.drop(['FE', 'SiO2', 'PO4', 'TDS', 'Turbidity', '%Na', 'Arsenic', 'LR. No'], axis=1, inplace=True)

columns = ['PH', 'EC', 'TH',
           'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'CARBONATE',
           'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE', 'SAR', 'RSC']

'''this part we fix the numrical columns'''
for col in columns:
    df[col].replace(['-', '<5', '<1', 'B', '*'], None, inplace=True)
    df[col], _ = find_and_replace_not_num_values(df[col], replace_to=None, inplace=True, astype=True, lops=True,
                                                 list_values=True)

df_get_dummies = pd.get_dummies(df.loc[:, ['SITE_TYPE', 'STATE_NAME']], columns=['SITE_TYPE', 'STATE_NAME'])
df_get_dummies = df_get_dummies.copy()
df_get_dummies.to_csv('Data/df_get_dummies.csv', index=False)

'''df ready to fill null '''
df_eda = df.copy()
df_eda.to_csv('Data/df_eda.csv', index=False)

'''now we can see statistical numeric columns'''
df_numeric = df.loc[:, 'PH':]
describe = df_numeric.describe()
corr = df_numeric.corr()
info_null = df.count()

'''this part shows that most of the SITE had less then 5 instance'''
# unique = unique_pd(df['SITE_ID'], df['SITE_ID'])
# unique = unique.drop('New')
# sns.histplot(unique)
# plt.show()

'''this part make histplot for all column'''
# cols = ['SITE_ID', 'WRIS ID', 'LATITUDE', 'LONGITUDE', 'SITE_TYPE',
# #        'STATE_NAME', 'DISTRICT_NAME', 'TAHSIL_NAME', 'BLOCK_NAME', 'SITE_NAME',
# #        'BASIN_NAME', 'PROJECT_NAME', 'year', 'PH', 'EC', 'TH',
#        'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE',
#        'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE']
# for col in tqdm(cols):
#     sns.histplot(df[col])
#     plt.show()


'''we can see that the amount of groundwater increases over time '''
unique_pd(df.groupby('WRIS ID').count()['year'][1:])

'''we sink to add new feature: fluoride t-1, so hear we can swo that no '''
df_id_per_year = df.loc[:, ['WRIS ID', 'year', 'FLUORIDE']]
df_id_per_year_sort = df_id_per_year.sort_values(['year', 'WRIS ID'])
df_id_per_year_std = df_id_per_year.groupby(['WRIS ID']).std()

'''we can see the difference between fluoride groundwater in other states '''

df_split_column_by_column_count = df_split_column_by_column(df=df, split_column='STATE_NAME', by_column='year')
