import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# from EDA import df_r
from func import unique_pd, find_and_replace_not_num_values, isfloat
path = r'df_r.csv'
data = pd.read_csv(path, low_memory=False)


df_num = data.loc[:, ['FLUORIDE', 'PH', 'EC', 'CA', 'NA', 'CHLORIDE', 'TH', 'BICARBONATE', 'NITRATE', 'FE', 'RSC', 'TOT_ALKALINITY', 'CARBONATE', 'SAR', 'SULPHATE', 'K', 'TDS', 'SiO2', 'Arsenic', 'PO4']]

class PredictNull:
    def __init__(self, series, df):
        self.df = df
        self.df = self.df[~self.df[series].isna()]
        # self.df = self.df.fillna(0.0000000001)
        self.X = self.df.drop([series], axis=1)
        self.X = pd.DataFrame(np.nan_to_num(self.X), columns=list(self.X.columns))
        self.series = series
        self.y = self.df[series]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def predict(self, regressor):
        print(f'predict { self.series}')
        self.regressor = regressor
        print(self.regressor.score(self.X_test, self.y_test))
        self.prediction = self.regressor.predict(self.X_test)
        unique_pd(pd.Series(self.prediction))
        self.unq = unique_pd(pd.Series(self.prediction))
        print(self.unq)

    def fill_null(self):
        for i in tqdm(range(0, len(self.y))):
            if list(self.df[self.series][i:i + 1].isna())[0] is True:
                self.y[i] = self.regressor.predict(self.X[i:i + 1])
        self.df_1 = self.df.copy()
        self.df_1.to_csv(f'df_fill_{self.series}.csv', index=False)




# df_num = df_num.fillna(0)
# # df_num = pd.DataFrame(np.nan_to_num(df_num), columns =['FLUORIDE','PH'] )
# X = df_num.loc[:, ['PH', 'FLUORIDE', 'CA', 'NA', 'CHLORIDE', 'TH', 'BICARBONATE', 'NITRATE', 'FE', 'RSC', 'TOT_ALKALINITY', 'CARBONATE', 'SAR', 'SULPHATE', 'K', 'TDS', 'SiO2', 'Arsenic', 'PO4']]
# y = df_num['EC']
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#
# '''Gradient Boosting Regressor to df['EC'] '''
# print('start')
# # GradientBoosting = GradientBoostingRegressor(n_estimators=80, learning_rate=0.1, random_state=0)
# # parameters = {'max_depth':(1, 2, 3, 4, 5), 'learning_rate':(0.5, 0.1, 0.05, 0.01), 'n_estimators':(60, 70, 75, 80, 85, 90 ,100)}
# # clf = GridSearchCV(GradientBoosting, parameters)
# clf = GradientBoostingRegressor(n_estimators=85, learning_rate=0.05, max_depth=4, random_state=42)
# clf.fit(X_train, y_train)
#
# print(clf.score(X_test, y_test))
# prediction = clf.predict(X_test)
# unique_pd(pd.Series(prediction))
# unq = unique_pd(pd.Series(prediction))
# print(unq)
#
# '''clf.best_params_:GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 85}'''
#
# for i in range(0, len(df_num['EC'])):
#     if list(df_num['EC'][i:i+1].isna())[0] is True:
#         df_num['EC'][i] = clf.predict(X[i:i+1])
# df_1 = df_num.copy()
# df_1.to_csv('df_1.csv', index=False)


if __name__ == '__main__':

    '''Regressor df['PH']'''
    # df_num['PH'] = df_num['PH'][df_num['PH']>0]
    # df_num['PH'] = df_num['PH'][df_num['PH']<20]
    # self = PredictNull(series='PH', df=df_num)
    '''Gradient Boosting Regressor to df['PH'] '''

    # ph_fill = PredictNull(series='PH', df=df_num)
    #
    # print('start')
    # GradientBoosting = GradientBoostingRegressor(random_state=0)
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_gb = GridSearchCV(GradientBoosting, parameters)
    # clf_gb.fit(ph_fill.X_train, ph_fill.y_train)
    # ph_fill.predict(clf_gb)
    #
    # AdaBoostRegressor = AdaBoostRegressor(random_state=0)
    # parameters = { 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_ab = GridSearchCV(GradientBoosting, parameters)
    # clf_ab.fit(ph_fill.X_train, ph_fill.y_train)
    # ph_fill.predict(clf_ab)
    #
    # RandomForestRegressor = RandomForestRegressor(random_state=0)
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_rf = GridSearchCV(GradientBoosting, parameters)
    # clf_rf.fit(ph_fill.X_train, ph_fill.y_train)
    # ph_fill.predict(clf_rf)

    # clf = GradientBoostingRegressor(n_estimators=85, learning_rate=0.05, max_depth=4, random_state=42)
    # clf = AdaBoostRegressor(n_estimators=85, learning_rate=0.05, random_state=42)
    # clf = RandomForestRegressor(n_estimators=85, max_depth=4, random_state=42)


    # clf.fit(ph_fill.X_train, ph_fill.y_train)
    # self.predict(clf)


    '''Gradient Boosting Regressor to df['EC'] '''

    # ec_fill = PredictNull(series='EC', df=df_num)
    #
    # print('start')
    # GradientBoosting = GradientBoostingRegressor(random_state=0)
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_gb = GridSearchCV(GradientBoosting, parameters)
    # clf_gb.fit(ec_fill.X_train, ec_fill.y_train)
    # ec_fill.predict(clf_gb)
    #
    # AdaBoostRegressor = AdaBoostRegressor(random_state=0)
    # parameters = { 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_ab = GridSearchCV(GradientBoosting, parameters)
    # clf_ab.fit(ec_fill.X_train, ec_fill.y_train)
    # ec_fill.predict(clf_ab)
    #
    # RandomForestRegressor = RandomForestRegressor(random_state=0)
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_rf = GridSearchCV(GradientBoosting, parameters)
    # clf_rf.fit(ec_fill.X_train, ec_fill.y_train)
    # ec_fill.predict(clf_rf)

    '''Gradient Boosting Regressor to df['CHLORIDE'] '''

    # chloride_fill = PredictNull(series='CHLORIDE', df=df_num)
    #
    # print('start')
    # GradientBoosting = GradientBoostingRegressor(random_state=0)
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_gb = GridSearchCV(GradientBoosting, parameters)
    # clf_gb.fit(chloride_fill.X_train, chloride_fill.y_train)
    # chloride_fill.predict(clf_gb)
    #
    # AdaBoostRegressor = AdaBoostRegressor(random_state=0)
    # parameters = { 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_ab = GridSearchCV(GradientBoosting, parameters)
    # clf_ab.fit(chloride_fill.X_train, chloride_fill.y_train)
    # chloride_fill.predict(clf_ab)
    #
    # RandomForestRegressor = RandomForestRegressor(random_state=0)
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # clf_rf = GridSearchCV(GradientBoosting, parameters)
    # clf_rf.fit(chloride_fill.X_train, chloride_fill.y_train)
    # chloride_fill.predict(clf_rf)


    '''Gradient Boosting Regressor to df['TH'] '''

    th_fill = PredictNull(series='TH', df=df_num)

    print('start GridSearchCV TH column')

    print('GradientBoosting')
    GradientBoosting = GradientBoostingRegressor(random_state=0)
    parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    clf_gb = GridSearchCV(GradientBoosting, parameters)
    clf_gb.fit(th_fill.X_train, th_fill.y_train)
    print(clf_gb.best_params_)
    th_fill.predict(clf_gb)

    print('AdaBoostRegressor')
    AdaBoostRegressor = AdaBoostRegressor(random_state=0)
    parameters = {'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    clf_ab = GridSearchCV(GradientBoosting, parameters)
    clf_ab.fit(th_fill.X_train, th_fill.y_train)
    print(clf_ab.best_params_)
    th_fill.predict(clf_ab)

    print('RandomForestRegressor')
    RandomForestRegressor = RandomForestRegressor(random_state=0)
    parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    clf_rf = GridSearchCV(GradientBoosting, parameters)
    clf_rf.fit(th_fill.X_train, th_fill.y_train)
    print(clf_rf.best_params_)
    th_fill.predict(clf_rf)




