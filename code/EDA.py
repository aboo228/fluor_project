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
        'SiO2', 'TDS', 'Turbidity', '%Na', 'Arsenic']

'''this part we fix th'''
columns = ['PH', 'EC', 'TH',
       'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE',
       'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE', 'SAR',
       'RSC', 'SiO2', 'PO4', 'TDS', 'Turbidity', '%Na', 'Arsenic']



for col in columns:
    print(col)
    df[col].replace(['-', '<5', '<1', 'B', '*'], None, inplace=True)

    df[col], _ = find_and_replace_not_num_values(df[col], replace_to=None, inplace=True, astype=True, lops=True, list_values=True)

col_to_fix = ['LATITUDE', 'LONGITUDE', 'LR. No', 'PO4', 'RSC']

'''fix df['RSC']'''
df['RSC'].replace([' ', '#REF!', '#VALUE!','ND'], None, inplace=True)
df['RSC'] = df['RSC'].astype('float32')

'''fix df['RSC']'''
df['PO4'].replace([' ', 'BDL', 'NIL', 'NS', 'Traces', 'leakage', 'nd', 'nil'], None, inplace=True)
df['PO4'].replace(['<0.01', '<0.1', '<0.10', '<0.11', '<0.12', '<0.13', '<0.14', '<0.15', '<0.16', '<0.17', '<0.18', '<0.19', '>0.10', '>0.11'], 0.2, inplace=True)
df['PO4'] = df['PO4'].astype('float32')


'''df ready to fill null '''
df_r = df.copy()
df_r.to_csv('df_.csv', index=False)

'''now we can look statistical numeric columns'''
df_numeric = df.loc[:, 'PH':'Arsenic']

describe = df_numeric.describe()
corr = df_numeric.corr()

'''this part make histplot for all column'''
# cols = ['PH', 'EC', 'TH',
#        'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE',
#        'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE', 'SAR',
#        'RSC', 'SiO2', 'PO4', 'TDS', 'Turbidity', '%Na', 'Arsenic']
# for col in tqdm(cols):
#     sns.histplot(df[col])
#     plt.show()

# info = df.count()
# info.to_excel("info.xlsx")


    df[col], _ = find_and_replace_not_num_values(df[col], inplace=True, astype=True, lops=True, list_values=True)

col_to_fix = ['LATITUDE', 'LONGITUDE', 'LR. No']


series = df['RSC']
list_str_unique_values = unique_pd(series[series.str.contains('-', na=False)]).index.to_list()
list_for_comparison = []
count = 0
while len(list_str_unique_values) != list_for_comparison:
    list_for_comparison = len(list_str_unique_values)
    try:
        for i in range(0, len(list_str_unique_values)):
            # print(list_str_unique_values[i])
            if isfloat(list_str_unique_values[i]) is True:
                list_str_unique_values.pop(i)
    except:
        'list index out of range'
