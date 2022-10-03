import pandas as pd
import numpy as np
from EDA import df

# todo to unite duplicate name in row - df['SITE_TYPE']
# SITE_TYPE_unique =  unique_pd(df['SITE_TYPE'], df['SITE_TYPE'])

# todo when value = New replace to 1
# SITE_ID_unique = unique_pd(df['SITE_ID'], df['SITE_ID'])

# todo distribution of df['PH']

# s = df['PH'][df['PH']>0]
# s = s[s<20]
# sns.histplot(s)
# plt.show()

# todo replace 'ND' in None - df['PH'], df['EC']

# df['EC'] = df['EC'].replace(['ND','LEAKED','leakage'],0)
# unique = unique_pd(df['EC'] )

# ec = df['EC'].astype('float32')
# ec.describe()

# todo replace 'ND','leakage'  in None - df['TH']

# unique_pd(df['TH'][df['TH'].fillna('ND').str.isalpha()])

# todo fix 'RSC' column, need  to find way to keep negative float numbers
# list_str_unique_values.drop(series[series.str.contains('-')].index, inplace=True)
#

# list_str_unique_value = list_str_unique_values.index.to_list()


# list_str_unique_value = list_str_unique_values.index.to_list()

