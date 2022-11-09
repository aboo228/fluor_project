import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from EDA import df_r
from func import unique_pd, find_and_replace_not_num_values, isfloat
path = r'Data/gdf.csv'
df = pd.read_csv(path, low_memory=False)
'''create dataset '''
df = df[~df['FLUORIDE'].isna()]
columns = ['PH', 'EC', 'TH',
       'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'CARBONATE',
       'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'SAR', 'RSC', 'PET', 'clay_content_1.5m', 'calcisols'
       ,'ardity', 'alpha', 'AET', 'tri', 'slopes', 'silt_content_1.5m',
          'sand_content_1.5m', 'precipitation std', 'precipitation mean', 'pH_content_1.5m', 'FLUORIDE']

df_fluoride = df.loc[:, columns]
df_fluoride = df_fluoride.fillna(0)
columns.remove('FLUORIDE')

X = df_fluoride.loc[:, columns]
y = df_fluoride['FLUORIDE'].copy()

'''convert target to boolean value'''
y[y <= 0.7] = 0
y[y.between(0.7, 2, inclusive='right')] = 1
y[y > 2] = 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 )






print('start')
'''GridSearchCV'''
# RandomForest = RandomForestClassifier(random_state=0)
# GradientBoosting = GradientBoostingClassifier( random_state=42)
# parameters = {'max_depth': (2, 3, 4, 5), 'n_estimators': (60, 80, 100, 150, 200, 250)}
# clf = GridSearchCV(RandomForest, parameters, verbose=3)

# clf = RandomForestClassifier(max_depth=4, n_estimators=150, random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
a = clf.predict(X_test)
unique_pd(pd.Series(a))
unq = unique_pd(pd.Series(a))
print(unq)


'''clf.best_params_ to GradientBoosting
Out[4]: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}'''


'''Feature importance'''
feature_names = X_train.columns.to_list()
forest = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
forest.fit(X_train, y_train)

import time

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# plt.show()
