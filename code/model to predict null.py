import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from func import unique_pd, find_and_replace_not_num_values, isfloat
path = r'Data/df_fill_NA.csv'
data = pd.read_csv(path, low_memory=False)

df_num = data.loc[:, ['PH', 'EC', 'TH',
       'TOT_ALKALINITY', 'CA', 'MG', 'NA', 'K', 'CARBONATE',
       'BICARBONATE', 'CHLORIDE', 'SULPHATE', 'NITRATE', 'FLUORIDE', 'SAR', 'RSC']]


class PredictNull:
    def __init__(self, series, df):
        self.df = df
        self.df = self.df.fillna(0.0000000001)
        self.df_train = self.df[~self.df[series].isna()]
        self.X = self.df_train.drop([series], axis=1)
        self.X = pd.DataFrame(np.nan_to_num(self.X), columns=list(self.X.columns))
        self.series = series
        self.y = self.df_train[series]
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

        self.regressor = GridSearchCV(model, parameters, verbose=3)
        self.regressor.fit(self.X_train, self.y_train)
        print(f'best_params for {self.series} column in {self.model} {self.regressor.best_params_}')

        if predict == True:
            self.predict()

        return model

    def predict(self, regressor=0):
        if regressor != 0:
            self.regressor = regressor
        print(f'prediction of { self.series} {self.model}')
        print(self.regressor.score(self.X_test, self.y_test))
        self.prediction = self.regressor.predict(self.X_test)
        unique_pd(pd.Series(self.prediction))
        print(self.prediction)

        ####for classification
        self.unq = unique_pd(pd.Series(self.prediction))


    def fill_null(self):
        for i in tqdm(range(0, len(self.df[self.series]))):
            if list(self.df[self.series][i:i + 1].isna())[0] is True:
                self.df[self.series][i] = self.regressor.predict(self.df[i:i + 1])
        self.df_1 = self.df.copy()
        self.df_1.to_csv(f'Data/df_fill_{self.series}.csv', index=False)


if __name__ == '__main__':

    '''predict to df['TH'] '''
    # th_fill = PredictNull(series='TH', df=df_num)

    # parameters = {'max_depth': (1, 2, 3, 4, 5), 'learning_rate': (0.5, 0.1, 0.05, 0.01), 'n_estimators': (60, 70, 75, 80, 85, 90, 100)}
    # th_reg_gb = th_fill.grid_search_cv('GradientBoostingRegressor', parameters, predict=True)

    # parameters = {'learning_rate': (0.5, 0.1, 0.05), 'n_estimators': (60, 70, 80, 90, 100)}
    # th_reg_ab = th_fill.grid_search_cv('AdaBoostRegressor', parameters, predict=True)
    #
    # parameters = {'max_depth': (3, 4, 5), 'n_estimators': (60, 70, 80, 90, 100)}
    # th_reg_rf = th_fill.grid_search_cv('RandomForestRegressor', parameters, predict=True)


    '''fill null in TH column'''
    # # {'learning_rate': 0.5, 'max_depth': 4, 'n_estimators': 100}
    # th_fill = PredictNull(series='TH', df=df_num)
    # # 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 250
    # regressor = GradientBoostingRegressor(learning_rate=0.5, max_depth=4, n_estimators=100, random_state=0)
    #
    # regressor.fit(th_fill.X_train, th_fill.y_train)
    # th_fill.predict(regressor=regressor)
    # th_fill.fill_null()
