import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from func import unique_pd, find_and_replace_not_num_values, isfloat

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

# df = df[df['STATE_NAME'] == 'Rajasthan']

df = df[~df['FLUORIDE'].isna()]

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print(unique_pd(self.y_test))


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

    def fit_and_predict(self, clf):
        self.clf = clf
        self.clf.fit(self.X_train, self.y_train)
        # print(clf.score(self.X_test, self.y_test))
        self.y_pred = self.clf.predict(self.X_test)
        unique_pd(pd.Series(self.y_pred))
        unique_y_pred = unique_pd(pd.Series(self.y_pred))
        print(f'{unique_y_pred[0]} under the threshold\n{unique_y_pred[1]} above the threshold')

    def confusion_matrix(self):

        self.confusionMatrix = confusion_matrix(self.y_test, self.y_pred)
        true_positive, false_positive, false_negative, true_negative = self.confusionMatrix[0, 0],\
                                                                       self.confusionMatrix[0, 1],\
                                                                       self.confusionMatrix[1, 0],\
                                                                       self.confusionMatrix[1, 1]

        recall = true_positive / (true_positive + false_positive)
        precision = true_positive / (true_positive + false_positive)
        accuracy = (true_positive + true_negative) / self.y_test.count()
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        print(f'recall is {"{:.2%}".format(recall)}\nprecision is {"{:.2%}".format(precision)}'
              f'\naccuracy is {"{:.2%}".format(accuracy)}\nSensitivity is {"{:.2%}".format(sensitivity)}'
              f'\nSpecificity is {"{:.2%}".format(specificity)}')
        print(f'\n{self.confusionMatrix}')

    def confusion_df(self, slice_by_feature, to_print=False):
        df_confusion = pd.concat([pd.Series(self.y_pred), self.y_test.reset_index()], axis=1)
        df_confusion['absulot erorr'] = (df_confusion[0] - df_confusion['FLUORIDE']).abs()
        confusion_list = df_confusion[df_confusion['absulot erorr'] == 1]['index'].to_list()

        path_data = r'Data/df_eda.csv'
        df_eda = pd.read_csv(path_data, low_memory=False)
        df_slice_by_index = df_eda.loc[df_confusion['index'].to_list()]
        series_slice_by_confusion_list = unique_pd(df_slice_by_index.loc[confusion_list][slice_by_feature])
        series_percentage = series_slice_by_confusion_list / unique_pd(df_slice_by_index[slice_by_feature])
        _ = pd.concat([pd.DataFrame(columns=['_']), series_percentage], axis=1)
        _.rename(columns={"STATE_NAME": "percentage"}, inplace=True)
        series_percentage = _.drop(['_'], axis=1)

        self.confusion_df_percentage = pd.concat([series_percentage, unique_pd(df_slice_by_index[slice_by_feature])], axis=1)
        self.confusion_df_percentage.rename(columns={"STATE_NAME": "Count"}, inplace=True)

        if to_print is True:
            print(self.confusion_df_percentage)



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
    GradientBoosting = GradientBoostingClassifier( random_state=42)
    parameters = {'max_depth': (2, 3, 4, 5), 'n_estimators': (60, 80, 100, 150, 200, 250)}
    clf = GridSearchCV(RandomForest, parameters, verbose=3)


    '''clf.best_params_ to GradientBoosting {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}'''
    clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    clf_AdaBoost = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf_RandomForest = RandomForestClassifier(max_depth=4, n_estimators=150, random_state=0)
    clf_LogisticRegression = LogisticRegression(random_state=0)
    clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))



    self = ClassificationModel(df)
    self.split_df_to_train_test(0.7)
    self.fit_and_predict(clf_GradientBoosting)
    self.confusion_matrix()
    self.confusion_df('STATE_NAME')

    # list_clf = [clf_GradientBoosting, clf_AdaBoost, clf_RandomForest, clf_LogisticRegression, clf_svm]
    # for clf in list_clf:
    #     print(clf)
    #     self = ClassificationModel(df)
    #     self.split_df_to_train_test(0.7)
    #     self.fit_and_predict(clf)
    #     self.confusion_matrix()
    #     self.confusion_df('STATE_NAME', to_print=True)





