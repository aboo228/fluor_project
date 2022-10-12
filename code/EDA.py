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


'''now we can look statistical numeric columns'''
describe = df.loc[:, 'PH':'Arsenic'].describe()

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

# df_numeric = df.loc[:, 'PH':'Arsenic']
# df_numeric.corr()


import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df_test = df.copy()

df_test = df.loc[:, ['FLUORIDE', 'PH', 'EC', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE']]
df_test = df_test.fillna(0)
# df_test = pd.DataFrame(np.nan_to_num(df_test), columns =['FLUORIDE','PH'] )
X = df_test.loc[:, ['PH', 'EC', 'CA', 'MG', 'NA', 'K', 'FE', 'CARBONATE']]
y = df_test['FLUORIDE']
param = 0.5
y[y > 0.7] = 1
y[y <= 0.7] = 0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 )

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
# clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)

print(clf.score(X_test, y_test))
a = clf.predict(X_test)
unique_pd(pd.Series(a))