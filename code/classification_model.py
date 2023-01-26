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
from sklearn.preprocessing import MinMaxScaler

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
# df = df[df['FLUORIDE'] > 0.01]  # remove the outliers



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

    def split_df_to_train_test(self, threshold_a, threshold_b=None, scale_up=None,technique_scale=None):

        # Assign threshold values to class variables
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b

        # Copy selected columns from dataframe to create 'X' and 'y'
        self.X = self.df.loc[:, 'AET.tif':].copy()
        self.X = self.X.drop(['FLUORIDE'], axis=1).fillna(0)
        self.y = self.df['FLUORIDE'].copy()

        # Convert target column 'y' to boolean values using threshold_a
        self.y[self.y <= threshold_a] = 0
        self.y[self.y > threshold_a] = 1
        self.y = self.y.astype('int')

        # Split dataframe into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=0.2, random_state=42,
                                                                                stratify=self.y, shuffle=True)
        if technique_scale == 'standard':
            self.X_train = StandardScaler().fit_transform(self.X_train)
            self.X_test = StandardScaler().fit_transform(self.X_test)
        elif technique_scale == 'minmax':
            self.X_train = MinMaxScaler().fit_transform(self.X_train)
            self.X_test = MinMaxScaler().fit_transform(self.X_test)
        else:
            pass

        if scale_up is not None:
            self.X_train, self.y_train = self.scaling_up(technique=scale_up)


        return self.X_train, self.X_test, self.y_train, self.y_test

    def scaling_up(self, technique='SMOTE'):
        from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
        from imblearn.under_sampling import RandomUnderSampler, NearMiss
        X_train, y_train = self.X_train, self.y_train

        if technique == 'SMOTE':
            sm = SMOTE(random_state=12)
            X_res, y_res = sm.fit_resample(X_train, y_train)
        elif technique == 'RandomUnderSampler':
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train)
        elif technique == 'NearMiss':
            nr = NearMiss(version=2, n_neighbors=3, n_neighbors_ver3=3, random_state=42)
            X_res, y_res = nr.fit_resample(X_train, y_train)
        elif technique == 'RandomOverSampler':
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X_train, y_train)
        elif technique == 'AdaptiveSynthetic':
            adasyn = ADASYN(random_state=42)
            X_res, y_res = adasyn.fit_resample(X_train, y_train)
        else:
            raise ValueError(
                "Invalid technique. Please choose from 'SMOTE', 'RandomUnderSampler', 'NearMiss', 'RandomOverSampler' , 'AdaptiveSynthetic'")
        return X_res, y_res

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
        counts = pd.Series(self.y_pred).value_counts()
        if len(counts) > 1:
            print(f'{counts[0]} under the threshold\n{counts[1]} above the threshold\n')
            self.X_test = X
        elif len(counts) == 1:
            return int(self.y_pred)
        return self.y_pred

    def confusion_matrix(self, y, confusionMatrix=None):  # calculate_confusion_matrix
        self.y_test = y
        if confusionMatrix is None:
            self.confusionMatrix = confusion_matrix(self.y_test, self.y_pred)
        else:
            self.confusionMatrix = confusionMatrix

        tp, fp, fn, tn = self.confusionMatrix.ravel()

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / self.y_test.count()
        sensitivity = recall
        specificity = tn / (tn + fp)
        f1_score = 2 * (specificity * recall) / (specificity + recall)

        matrix = {'recall': recall, 'precision': precision, 'accuracy': accuracy, 'sensitivity': sensitivity,
                  'specificity': specificity, 'f1_score': f1_score}
        matrix = pd.DataFrame.from_dict(matrix, orient='index').T
        self.print_metrics(matrix)

        return matrix, self.confusionMatrix

    def print_metrics(self, matrix):
        print(
            f'recall is {"{:.2%}".format(matrix["recall"][0])}'
            f'\nprecision is {"{:.2%}".format(matrix["precision"][0])}'
            f'\naccuracy is {"{:.2%}".format(matrix["accuracy"][0])}'
            f'\nSensitivity is {"{:.2%}".format(matrix["sensitivity"][0])}'
            f'\nSpecificity is {"{:.2%}".format(matrix["specificity"][0])}'
            f'\nF1 score is {"{:.2%}".format(matrix["f1_score"][0])}')
        print(f'\n{self.confusionMatrix}')

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
        self.confusion_df_percentage.sort_values(by=['P of confusion'], inplace=True)



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
    # RandomForest = RandomForestClassifier(random_state=0)
    # GradientBoosting = GradientBoostingClassifier(random_state=42)
    # parameters = {'max_depth': (2, 3, 4, 5), 'n_estimators': (60, 80, 100, 150, 200, 250)}
    # clf = GridSearchCV(RandomForest, parameters, verbose=3)



    clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    self = ClassificationModel(df)
    self.split_df_to_train_test(0.7)
    self.fit(clf_GradientBoosting)
    self.predict(self.X_test)
    self.confusion_matrix(self.y_test)
    self.confusion_df('STATE_NAME', to_print=True)

'''chack scale up techniques'''
    # techniques = ['SMOTE', 'RandomUnderSampler', 'RandomOverSampler', 'AdaptiveSynthetic']
    # list_cun = []
    # list_cun.append(self.confusion_df_percentage['P of confusion'].rename('P basic'))
    # list_cun.append(self.confusion_df_percentage['Count state'].rename('Count basic'))
    # for  technique in techniques:
    #
    #     self.split_df_to_train_test(0.7, scale_up=technique)
    #     self.fit(clf_GradientBoosting)
    #     self.predict(self.X_test)
    #     self.confusion_matrix(self.y_test)
    #     _ = self.confusion_df('STATE_NAME')
    #     list_cun.append(_['P of confusion'].rename(technique))
    # df_scale_up = pd.concat(list_cun, axis=1)

def compare_model(self, clf, to_print=False):
    self.fit(clf)
    self.predict(self.X_test)
    self.confusion_matrix(self.y_test)
    self.confusion_df('STATE_NAME', to_print=to_print)
    return self.confusion_df_percentage



# create function to compare between dataframe of confusion_df_percentage the function gets list_of_confusion_df for all clf, the function return 2 dataframe
# one dataframe concat the "Count confusion state" columns and  the second concat"P of confusion" columns

def compare_confusion_df(list_of_confusion_df, classifiers_name):
    list_of_confusion_df = [df for df in list_of_confusion_df]
    df_count_confusion_state = pd.concat([df['Count confusion state'] for df in list_of_confusion_df], axis=1)
    df_count_confusion_state.columns = [f'Count confusion state {clf}' for clf in classifiers_name]
    df_p_confusion = pd.concat([df['P of confusion'] for df in list_of_confusion_df], axis=1)
    df_p_confusion.columns = [f'P of confusion {clf}' for clf in classifiers_name]
    return df_count_confusion_state.T, df_p_confusion.T

def define_clf():
    clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    clf_AdaBoost = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf_RandomForest = RandomForestClassifier(max_depth=4, n_estimators=150, random_state=0)
    clf_ExtraTrees = ExtraTreesClassifier(max_depth=4, n_estimators=150, random_state=0)
    clf_LogisticRegression = LogisticRegression(random_state=0)
    clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_xgb = XGBClassifier(n_estimators=100, max_depth=5, eta=0.1)
    list_of_Classifiers = [clf_GradientBoosting, clf_AdaBoost, clf_RandomForest, clf_ExtraTrees, clf_xgb]
    classifiers_name = ["clf_GradientBoosting", "clf_AdaBoost", "clf_RandomForest", "clf_ExtraTrees", "clf_xgb"]
    return list_of_Classifiers, classifiers_name

# list_of_Classifiers, classifiers_name = define_clf()
# list_of_confusion_df = []
# for i in list_of_Classifiers:
#     _ = compare_model(self, i)
#     list_of_confusion_df.append(_.copy())
# df_count_confusion_state, df_p_confusion = compare_confusion_df(list_of_confusion_df, classifiers_name)
# print(df_count_confusion_state, df_p_confusion)
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
