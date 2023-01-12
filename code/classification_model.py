import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from func import *

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

'''SVM classifier'''
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

path = r'Data/gdf.csv'
df = pd.read_csv(path, low_memory=False)

'''df_get_dummies SITE_TYPE, STATE_NAME'''
path_df_get_dummies = 'Data/df_get_dummies.csv'
df_get_dummies = pd.read_csv(path_df_get_dummies, low_memory=False)

df = pd.concat([df, df_get_dummies], axis=1)

''' Division by regions'''
country_list_of_dakan = ['Andhra Pradesh', 'Dadra And Nagar Haveli', 'Goa', 'Karnataka',
                         'Kerala', 'Maharashtra', 'Odisha', 'Pondicherry', 'Tamil Nadu', 'Telangana', ]
country_list_of_himalayan = ['Arunachal Pradesh', 'Assam', 'Himachal Pradesh', 'Jammu And Kashmir',
                             'Meghalaya', 'Nagaland', 'Tripura', 'Uttarakhand', ]
country_list_of_lowland = ['Bihar', 'Chhattisgarh', 'Delhi', 'Gujarat', 'Haryana', 'Jharkhand',
                           'Punjab', 'Rajasthan', 'Uttar Pradesh', 'West Bengal']

df_dakan = df.query(f'STATE_NAME == {country_list_of_dakan}')
df_himalayan = df.query(f'STATE_NAME == {country_list_of_himalayan}')
df_lowland = df.query(f'STATE_NAME == {country_list_of_lowland}')
df_list = [df_dakan, df_himalayan, df_lowland]
# df = df_dakan
# df = df_himalayan
# df = df_lowland


df = df[df['FLUORIDE'] < 30]  # remove the outliers


# df = df[~df['FLUORIDE'].isna()]

class ClassificationModel:
    def __init__(self, data):
        self.df = data
        self.X, self.y = None, None
        self.threshold_a, self.threshold_b = None, None
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_pred = None, None, None, None, None
        self.clf = None
        self.confusionMatrix = None
        self.confusion_df_percentage = None

    def split_df_to_train_test(self, threshold_a, threshold_b=None):
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b

        self.X = self.df.loc[:, 'AET.tif':].copy()
        self.X = self.X.drop(['FLUORIDE'], axis=1).fillna(0)
        self.y = self.df['FLUORIDE'].copy()

        '''convert target to boolean value'''
        self.y[self.y <= self.threshold_a] = 0
        self.y[self.y > self.threshold_a] = 1
        # self.y[self.y.between(self.threshold_a, self.threshold_b, inclusive='right')] = 1
        # self.y[self.y > 2] = self.threshold_b
        self.y = self.y.astype('int')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.2, random_state=42,
                                                                                stratify=self.y, shuffle=True)
        print(unique_pd(self.y_test))
        return self.X_train, self.X_test, self.y_train, self.y_test

    def split_df_by_state(self, state, threshold_a):
        '''
        split to train and test by state
        '''
        self.threshold_a = threshold_a
        train = self.df[self.df['STATE_NAME'] != state]
        test = self.df[self.df['STATE_NAME'] == state]

        self.X_train = train.drop(['FLUORIDE'], axis=1).copy()
        self.y_train = train['FLUORIDE'].copy()
        self.X_test = test.drop(['FLUORIDE'], axis=1).copy()
        self.y_test = test['FLUORIDE'].copy()
        self.X_train = self.X_train.loc[:, 'AET.tif':]
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.loc[:, 'AET.tif':]
        self.X_test = self.X_test.fillna(0)
        '''convert target to boolean value'''
        self.y_train[self.y_train <= self.threshold_a] = 0
        self.y_train[self.y_train > self.threshold_a] = 1
        self.y_test[self.y_test <= self.threshold_a] = 0
        self.y_test[self.y_test > self.threshold_a] = 1

    def fit(self, clf):
        self.clf = clf
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, X):
        self.y_pred = self.clf.predict(X)
        unique_pd(pd.Series(self.y_pred))
        if len(X) > 1:
            unique_y_pred = unique_pd(pd.Series(self.y_pred))
            print(f'{unique_y_pred[0]} under the threshold\n{unique_y_pred[1]} above the threshold')
            self.X_test = X
        elif len(X) == 1:
            print(self.y_pred)
            return int(self.y_pred)

    def confusion_matrix(self, y, confusionMatrix=None):
        self.y_test = y

        if confusionMatrix is None:
            self.confusionMatrix = confusion_matrix(self.y_test, self.y_pred)
        else:
            self.confusionMatrix = confusionMatrix
        true_positive, false_positive, false_negative, true_negative = self.confusionMatrix[0, 0], \
                                                                       self.confusionMatrix[0, 1], \
                                                                       self.confusionMatrix[1, 0], \
                                                                       self.confusionMatrix[1, 1]

        recall = true_positive / (true_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        accuracy = (true_positive + true_negative) / self.y_test.count()
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        print(f'recall is {"{:.2%}".format(recall)}\nprecision is {"{:.2%}".format(precision)}'
              f'\naccuracy is {"{:.2%}".format(accuracy)}\nSensitivity is {"{:.2%}".format(sensitivity)}'
              f'\nSpecificity is {"{:.2%}".format(specificity)}')
        print(f'\n{self.confusionMatrix}')
        list_matrix = [recall, precision, accuracy, sensitivity, specificity]
        matrix = {'recall': recall, 'precision': precision,
                  'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity}
        matrix = pd.DataFrame.from_dict(matrix, orient='index').T
        return matrix, self.confusionMatrix

    def confusion_df(self, slice_by_feature, to_print=False):
        df_confusion = pd.concat([pd.Series(self.y_pred), self.y_test.reset_index()], axis=1)
        df_confusion['absulot erorr'] = (df_confusion[0] - df_confusion['FLUORIDE']).abs()
        confusion_list = df_confusion[df_confusion['absulot erorr'] == 1]['index'].to_list()

        path_data = r'Data/df_eda.csv'
        df_eda = pd.read_csv(path_data, low_memory=False)
        df_slice_by_index = df_eda.loc[df_confusion['index'].to_list()]
        series_slice_by_confusion_list = unique_pd(df_slice_by_index.loc[confusion_list][slice_by_feature])
        series_percentage = (series_slice_by_confusion_list / unique_pd(df_slice_by_index[slice_by_feature])).round(2)
        _ = pd.concat([pd.DataFrame(columns=['_']), series_percentage], axis=1)
        _.rename(columns={"STATE_NAME": "P of confusion"}, inplace=True)
        series_percentage = _.drop(['_'], axis=1)
        series_percentage = pd.concat([series_percentage, series_slice_by_confusion_list], axis=1)
        series_percentage.rename(columns={"STATE_NAME": "Count confusion state"}, inplace=True)

        self.confusion_df_percentage = pd.concat([series_percentage, unique_pd(df_slice_by_index[slice_by_feature])],
                                                 axis=1)
        self.confusion_df_percentage.rename(columns={"STATE_NAME": "Count state"}, inplace=True)

        if to_print is True:
            print(self.confusion_df_percentage)
        return self.confusion_df_percentage

    def Feature_importance(self):
        import time

        feature_names = self.X_train.columns.to_list()
        forest = RandomForestClassifier(max_depth=5, n_estimators=200, random_state=0)
        forest.fit(self.X_train, self.y_train)
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
        plt.show()


if __name__ == '__main__':
    print('start')

    '''GridSearchCV'''
    RandomForest = RandomForestClassifier(random_state=0)
    GradientBoosting = GradientBoostingClassifier(random_state=42)
    parameters = {'max_depth': (2, 3, 4, 5), 'n_estimators': (60, 80, 100, 150, 200, 250)}
    clf = GridSearchCV(RandomForest, parameters, verbose=3)

    '''list of Classifiers'''
    '''clf.best_params_ to GradientBoosting {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}'''

    clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    clf_AdaBoost = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf_RandomForest = RandomForestClassifier(max_depth=4, n_estimators=150, random_state=0)
    clf_ExtraTrees = ExtraTreesClassifier(max_depth=4, n_estimators=150, random_state=0)
    clf_LogisticRegression = LogisticRegression(random_state=0)
    clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_xgb = XGBClassifier(n_estimators=100, max_depth=5, eta=0.1)
    list_of_Classifiers = [clf_GradientBoosting, clf_AdaBoost, clf_RandomForest, clf_ExtraTrees, clf_LogisticRegression,
                           clf_svm, clf_xgb]

    self = ClassificationModel(df)
    self.split_df_to_train_test(0.7)
    # self.fit(clf_GradientBoosting)
    # self.predict(self.X_test)
    # self.confusion_matrix(self.y_test)
    # self.confusion_df('STATE_NAME', to_print=True)

list_of_confusion_df = []
for i in list_of_Classifiers:
    self.fit(clf_GradientBoosting)
    self.predict(self.X_test)
    self.confusion_matrix(self.y_test)
    _ = self.confusion_df('STATE_NAME', to_print=True)
    list_of_confusion_df.append(_.copy())

'''chek difrent classifier'''
# list_clf = [clf_GradientBoosting, clf_AdaBoost, clf_RandomForest, clf_LogisticRegression, clf_svm]
# for clf in list_clf:
#     print(clf)
#     self = ClassificationModel(df)
#     self.split_df_to_train_test(0.7)
#     self.fit_and_predict(clf)
#     self.confusion_matrix()
#     self.confusion_df('STATE_NAME', to_print=True)

'''comparision between basice model and partitioned model  '''
# confusion_matrix_list = []
# for dataframe in tqdm(df_list):
#     self = ClassificationModel(dataframe)
#     self.split_df_to_train_test(0.7)
#     self.fit(clf_GradientBoosting)
#     self.predict(self.X_test)
#     m, c = self.confusion_matrix(self.y_test)
#     self.confusion_df('STATE_NAME', to_print=True)
#     confusion_matrix_list.append(c)
#
# confusion_matrix_sum = confusion_matrix_list[0] + confusion_matrix_list[1] + confusion_matrix_list[2]
# self = ClassificationModel(df)
# self.split_df_to_train_test(0.7)
# m, c = self.confusion_matrix(self.y_test, confusion_matrix_sum)

'''runing lazypredict on data'''
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier

# self = ClassificationModel(df)
# X_train, X_test, y_train, y_test =self.split_df_to_train_test(0.7)

# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)
# model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
# clf_models = models
# clf_models


# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)
# model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
# reg_models = models
# reg_models
