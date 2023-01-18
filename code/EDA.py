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

replace_dict = {'RSC': [' ', '#REF!', '#VALUE!', 'ND'],
                'STATE_NAME': {'Uttarakhand ': 'Uttarakhand',
                               'Daman & Diu': 'Dadra And Nagar Haveli',
                               'Chandigarh': 'Punjab'},
                'SITE_TYPE': {'DW': 'Dug Well', 'D.W.': 'Dug Well', 'Dug well': 'Dug Well', ' Dug Well': 'Dug Well', 'dug': 'Dug Well', 'Dugwell': 'Dug Well',
                              'DUGWWELL': 'Dug Well', 'Dug': 'Dug Well', 'DUG': 'Dug Well', 'WELL': 'Dug Well', 'D/W': 'Dug Well', 'Dug Welll': 'Dug Well',
                              'Dug ': 'Dug Well', 'DUG WELL': 'Dug Well', 'DCB': 'Dug Well', 'Mark II ': 'Dug Well', 'CGWB SPZ': 'Dug Well',
                              'MONITORING': 'Monitoring', 'BOREWELL': 'Bore Well', 'BW': 'Bore Well', 'BW ': 'Bore Well', 'TW': 'Tube Well', 'T/W': 'Tube Well',
                              'CYL. TW': 'Tube Well', 'T.W': 'Tube Well', ' TW': 'Tube Well', 'H/P': 'Hand Pump', 'H.P': 'Hand Pump', 'HP': 'Hand Pump', 'Hand pump': 'Hand Pump',
                              'Mark II': 'Mark-II', 'markII': 'Mark-II', 'Cylindrical ': 'Cylindrical', 'PZ': 'Pz', 'STW': 'Sub T.W.', 'TW M-II': 'TW Mark II',
                              'Tw Mark-II': 'TW Mark II', 'Mark-II T.W.': 'TW Mark II', 'TW Mark-II': 'TW Mark II', 'T.W.': 'TW Mark II', 'Tw': 'TW Mark II',
                              'markII TW': 'TW Mark II', 'TW MK-II': 'TW Mark II', 'Twmark-II': 'TW Mark II', 'Mrk II': 'TW Mark II', 'Stw': 'TW Mark II',
                              'Submersible TW': 'TW Mark II', 'TW ': 'TW Mark II', 'Mark-II TW': 'TW Mark II', 'TW mark II': 'TW Mark II', 'T.W. Mark-II': 'TW Mark II',
                              'MARK-II': 'TW Mark II', 'Mark-II Tw': 'TW Mark II', 'Mark-Il': 'TW Mark II', 'SEW': 'TW Mark II', 'Sub T.W': 'TW Mark II', 'TIRUVALLUR': 'TW Mark II'}}

for col, replacement in replace_dict.items():
    if isinstance(replacement, dict):
        df[col].replace(replacement, inplace=True)
    else:
        df[col].replace(None, inplace=True)


high_null_cols = ['FE', 'SiO2', 'PO4', 'TDS', 'Turbidity', '%Na', 'Arsenic', 'LR. No']
df.drop(high_null_cols, axis=1, inplace=True)

num_cols = ['PH', 'EC', 'TH', 'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'CARBONATE', 'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE', 'SAR', 'RSC']
replace_values = ['-', '<5', '<1', 'B', '*']

df[num_cols] = df[num_cols].replace(replace_values, np.nan)
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')


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
