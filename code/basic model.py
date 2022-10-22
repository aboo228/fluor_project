import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from EDA import df_r
from func import unique_pd, find_and_replace_not_num_values, isfloat
path = r'df_1.csv'
df = pd.read_csv(path, low_memory=False)

df = df[~df['FLUORIDE'].isna()]

df_test = df.loc[:, ['FLUORIDE', 'PH', 'EC', 'CA', 'NA', 'CHLORIDE', 'TH', 'BICARBONATE', 'NITRATE', 'FE', 'RSC', 'TOT_ALKALINITY', 'CARBONATE', 'SAR', 'SULPHATE', 'K', 'TDS', 'SiO2', 'Arsenic', 'PO4']]
df_test = df_test.fillna(0)
# df_test = pd.DataFrame(np.nan_to_num(df_test), columns =['FLUORIDE','PH'] )
X = df_test.loc[:, ['PH', 'EC', 'CA', 'NA', 'CHLORIDE', 'TH', 'BICARBONATE', 'NITRATE', 'FE', 'RSC', 'TOT_ALKALINITY', 'CARBONATE', 'SAR', 'SULPHATE', 'K', 'TDS', 'SiO2', 'Arsenic', 'PO4']]
y = df_test['FLUORIDE']
'''convert target to boolean value'''
param = 0.5
y[y > 0.7] = 1
y[y <= 0.7] = 0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 )

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0).fit(X_train, y_train)
# clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)


'''GridSearchCV'''
print('start')
GradientBoosting = GradientBoostingClassifier( random_state=42)
# parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90 ,100)}
# clf = GridSearchCV(GradientBoosting, parameters)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
a = clf.predict(X_test)
unique_pd(pd.Series(a))
unq = unique_pd(pd.Series(a))
print(unq)


'''clf.best_params_
Out[4]: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}'''