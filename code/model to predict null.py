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
        self.model = None
        self.regressor = None
        self.prediction = None


    def grid_search_cv(self, model, parameters, predict=True):

        print(f'start GridSearchCV {self.series} column in {model}')
        self.model = model

        if model == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(random_state=0)

        elif model == 'AdaBoostRegressor':
            model = AdaBoostRegressor(random_state=0)

        elif model == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=0)

        self.regressor = GridSearchCV(model, parameters)
        self.regressor.fit(self.X_train, self.y_train)
        print(f'best_params for {self.series} column in {self.model} {self.regressor.best_params_}')

        if predict == True:
            self.predict()

        return model

    def predict(self):
        print(f'prediction of { self.series} {self.model}')
        print(self.regressor.score(self.X_test, self.y_test))
        self.prediction = self.regressor.predict(self.X_test)
        unique_pd(pd.Series(self.prediction))
        print(self.prediction)

        ####for classification
        self.unq = unique_pd(pd.Series(self.prediction))


    def fill_null(self):
        for i in tqdm(range(0, len(self.y))):
            if list(self.df[self.series][i:i + 1].isna())[0] is True:
                self.y[i] = self.regressor.predict(self.X[i:i + 1])
        self.df_1 = self.df.copy()
        self.df_1.to_csv(f'df_fill_{self.series}.csv', index=False)


# '''EC column : clf.best_params_:GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 85}'''
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
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # ph_reg_gb = ph_fill.grid_search_cv('GradientBoostingRegressor', parameters, predict=True)
    #
    # parameters = {'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # ph_reg_ab = ph_fill.grid_search_cv('AdaBoostRegressor', parameters, predict=True)
    #
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # ph_reg_rf = ph_fill.grid_search_cv('RandomForestRegressor', parameters, predict=True)
    #
    #
    #
    # '''Gradient Boosting Regressor to df['EC'] '''
    #
    # ec_fill = PredictNull(series='EC', df=df_num)
    #
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # ec_reg_gb = ec_fill.grid_search_cv('GradientBoostingRegressor', parameters, predict=True)
    #
    # parameters = {'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # ec_reg_ab = ec_fill.grid_search_cv('AdaBoostRegressor', parameters, predict=True)
    #
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # ec_reg_rf = ec_fill.grid_search_cv('RandomForestRegressor', parameters, predict=True)
    #
    #
    #
    # '''Gradient Boosting Regressor to df['CHLORIDE'] '''
    #
    # chloride_fill = PredictNull(series='CHLORIDE', df=df_num)
    #
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # chloride_reg_gb = chloride_fill.grid_search_cv('GradientBoostingRegressor', parameters, predict=True)
    #
    # parameters = {'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # chloride_reg_ab = chloride_fill.grid_search_cv('AdaBoostRegressor', parameters, predict=True)
    #
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # chloride_reg_rf = chloride_fill.grid_search_cv('RandomForestRegressor', parameters, predict=True)




    '''Gradient Boosting Regressor to df['TH'] '''

    th_fill = PredictNull(series='TH', df=df_num)

    parameters = {'max_depth': (1, 2), 'learning_rate': (0.5, 0.1), 'n_estimators': (60, 70)}
    th_reg_gb = th_fill.grid_search_cv('GradientBoostingRegressor', parameters, predict=True)

    # parameters = {'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # th_reg_ab = th_fill.grid_search_cv('AdaBoostRegressor', parameters, predict=True)
    #
    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # th_reg_rf = th_fill.grid_search_cv('RandomForestRegressor', parameters, predict=True)




