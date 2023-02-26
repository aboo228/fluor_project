import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from func import *
from classification_model import ClassificationModel
from NN_pytorch import Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
import json

class BagModel():
    def __init__(self, path_df, path_dict_country, df_for_test=False):
        self.df = self.load_df(path_df, df_for_test)
        df = self.df
        _, X_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        self.X_test = X_test.reset_index()
        self.y_test = self.X_test['FLUORIDE']
        self.create_dfs_india(path_dict_country)


    def load_df(self, path, df_for_test=False):
        self.original_df = pd.read_csv(path, low_memory=False)
        if df_for_test is True:
            _, self.df_short = train_test_split(self.original_df, test_size=0.1,
                                           random_state=42, shuffle=True, stratify=self.original_df['STATE_NAME'])
            self.df_short = self.df_short.reset_index(drop=True)
            self.df = self.df_short
        else:
            self.df = self.original_df

        self.df.query('FLUORIDE < 30', inplace=False)  # remove the outliers
        df_get_dummies = pd.get_dummies(self.df.loc[:, ['SITE_TYPE', 'STATE_NAME']], columns=['SITE_TYPE', 'STATE_NAME'])
        self.df_get_dummies = df_get_dummies.copy()
        self.df = pd.concat([self.df, self.df_get_dummies], axis=1)
        return self.df

    def create_dfs_india(self, dict_path):

        with open(dict_path, 'r') as file:
            json_str = file.read()
            self.dict_country_list = json.loads(json_str)

        df_dakan = self.df.query(f'STATE_NAME == {self.dict_country_list["dakan"]}')
        df_himalayan = self.df.query(f'STATE_NAME == {self.dict_country_list["himalayan"]}')
        df_lowland = self.df.query(f'STATE_NAME == {self.dict_country_list["lowland"]}')
        self.dfs_dict = {'basic': self.df, 'dakan': df_dakan, 'himalayan': df_himalayan, 'lowland': df_lowland}
        return df_dakan, df_himalayan, df_lowland, self.dfs_dict


    def load_clf(self, clf_model=None, nn_model=None, first_run=True, name_model='basic'):
        # self.clf_model = clf_model
        # self.nn_model = nn_model
        if first_run is True:
            clf_model = clf_model(self.df)
            nn_model = nn_model(self.df)
            clf_model.split_df_to_train_test(0.7)
            clf_model.fit(clf_GradientBoosting)
            nn_model.split_df_to_train_test(threshold_a=0.7, val=True)
            nn_model.fit(lr=0.004647, epochs=60)
            joblib.dump(clf_model, f'clf_model_{name_model}.joblib')
            joblib.dump(nn_model, f'nn_model_{name_model}.joblib')
        elif nn_model is None and clf_model is None:
            clf_model = joblib.load(f'clf_model_{name_model}.joblib')
            nn_model = joblib.load(f'nn_model_{name_model}.joblib')
        return nn_model, clf_model

    def predict_sample(self, sample, clf_model, nn_model):

        sample = self.convert_df(sample)
        clf_pred = clf_model.predict(sample)
        sample = torch.tensor(sample.values).type('torch.FloatTensor')
        nn_pred = nn_model.predict(sample)
        # todo look the return
        return [clf_pred, nn_pred], clf_model, nn_model


    def load_models(self, models):
        models = ['basic', 'dakan', 'himalayan', 'lowland']
        loaded_models = []
        for model in models:
            clf_model = joblib.load(f'clf_model_{model}.joblib')
            nn_model = joblib.load(f'nn_model_{model}.joblib')
            loaded_models.append(clf_model)
            loaded_models.append(nn_model)
        return loaded_models


    def generate_clf(self, dfs, load_models_flag=False):
        if load_models_flag:
            return self.load_models(dfs.keys())

        generated_models = []
        first_run = True
        name_basic_df = list(dfs.keys())[0]
        basic_df = dfs[name_basic_df]
        sample = dfs[name_basic_df].reset_index().loc[1:1, :]
        for model_name, first_model in dfs.items():
            _, clf_model, nn_model = self.predict_sample(sample, basic_df, ClassificationModel, Model, first_run,
                                                    name_model=model_name)
            generated_models.append(clf_model)
            generated_models.append(nn_model)
        return generated_models


    def predict_model(self):
        self.X_test_n = self.add_dummy_columns(self.X_test, self.dict_country_list, 'STATE_NAME')
        #functuon that recive df  and run all over the df whit loop, and return predict list
        clf_model, nn_model, clf_model_dakan, nn_model_dakan, clf_model_himalayan,\
        nn_model_himalayan, clf_model_lowland, nn_model_lowland = self.generate_clf(self.dfs_dict, load_models_flag=True)

        bacic, dakan, himalayan, lowland, other = 0, 0, 0, 0, 0
        first_run = False
        predictions = []

        for i in tqdm(range(len(self.X_test))):
            sample = self.X_test.loc[i:i, :]
            pred, _, _ = self.predict_sample(sample, clf_model, nn_model)

            if pred[0] == pred[1]:
                predictions.append(pred)
                bacic += 1
            elif self.X_test_n.loc[i:i, 'dakan'].any() == 1:
                dakan += 1
                pred, _, _ = self.predict_sample(sample, clf_model_dakan, nn_model_dakan)
                predictions.append(pred)
            elif (self.X_test_n.loc[i:i, 'himalayan'] == 1).any():
                himalayan += 1

                pred, _, _ = self.predict_sample(sample, clf_model_himalayan, nn_model_himalayan)
                predictions.append(pred)
            elif (self.X_test_n.loc[i:i, 'lowland'] == 1).any():
                lowland += 1
                pred, _, _ = self.predict_sample(sample, clf_model_lowland, nn_model_lowland)
                predictions.append(pred)
            else:
                other += 1
                predictions.append(pred)

        self.predict_accuracy(predictions, print_confusion_matrix=True)
        print(f'accuracy: {self.accuracy}')

    def predict_accuracy(self, predictions, print_confusion_matrix=False):
          # function that recive predict list and return accuracy
       y_pred = []
       for i in range(0, len(predictions)):
            y_pred.append(predictions[i][0])
       self.y_pred = np.array(y_pred)
       self.y_test = self.convert_to_binary(self.y_test, 0.7)
       self.accuracy = accuracy_score(self.y_test, self.y_pred)
       if print_confusion_matrix is True:
           matrix = self.creat_confusion_matrix()
           self.print_confusion_matrix(matrix)

       return self.accuracy

    def convert_to_binary(self, series, threshold):
        series[series <= threshold] = 0
        series[series > threshold] = 1
        series = series.astype('int')
        return series

    def add_dummy_columns(self, df, dict, column_name):
        dummy_df = pd.DataFrame(0, index=df.index, columns=dict.keys())

        for index, row in df.iterrows():
            value = row[column_name]
            for key in dict.keys():
                values = dict[key]
                if value in values:
                    dummy_df.at[index, key] = 1
        df = df.join(dummy_df)

        return df

    #todo
    def convert_df(self, df):
        X = df.loc[:, 'AET.tif':].copy()
        X = X.drop(['FLUORIDE'], axis=1).fillna(0)
        return X

    # function that recive recall,precision, f1 and accuracy specificity, and print them
    def creat_confusion_matrix(self, confusionMatrix=None):  # calculate_confusion_matrix
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
        # print_metrics(matrix)
        return matrix

    def print_confusion_matrix(self, matrix):
        print(
            f'recall is {"{:.2%}".format(matrix["recall"][0])}'
            f'\nprecision is {"{:.2%}".format(matrix["precision"][0])}'
            f'\naccuracy is {"{:.2%}".format(matrix["accuracy"][0])}'
            f'\nSensitivity is {"{:.2%}".format(matrix["sensitivity"][0])}'
            f'\nSpecificity is {"{:.2%}".format(matrix["specificity"][0])}'
            f'\nF1 score is {"{:.2%}".format(matrix["f1_score"][0])}')
        print(f'\n{self.confusionMatrix}')



path = r'Data/gdf.csv'
dict_country_path = 'Data/dict_country_list.json'


#todo: convert  country list to dict and crate func taht save and lode from os

''' Division by regions'''
clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

self = BagModel(path, dict_country_path, df_for_test=True)
self.predict_model()


