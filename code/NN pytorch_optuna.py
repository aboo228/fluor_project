import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from func import unique_pd, find_and_replace_not_num_values, isfloat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

path = r'Data/gdf.csv'
df = pd.read_csv(path, low_memory=False)

'''df_get_dummies SITE_TYPE, STATE_NAME'''
path_df_get_dummies = 'Data/df_get_dummies.csv'
df_get_dummies = pd.read_csv(path_df_get_dummies, low_memory=False)

df = pd.concat([df, df_get_dummies], axis=1)

# df = df[df['STATE_NAME'] == 'Rajasthan']

df = df[~df['FLUORIDE'].isna()]


class Model(nn.Module):
    def __init__(self, df, params):
        super().__init__()
        self.params = params
        self.df = df
        self.fc1 = nn.Linear(in_features=90, out_features=self.params['l1'])
        self.fc2 = nn.Linear(in_features=self.params['l1'], out_features=self.params['l2'])
        self.fc3 = nn.Linear(in_features=self.params['l2'], out_features=self.params['l3'])
        self.fc4 = nn.Linear(in_features=self.params['l3'], out_features=self.params['l4'])
        self.fc5 = nn.Linear(in_features=self.params['l4'], out_features=self.params['l5'])
        self.output = nn.Linear(in_features=self.params['l5'], out_features=2)

        self.dropout = nn.Dropout(p=params['dropout'])

    def forward(self, x=None):
        if x is None:
            x = self.X_train
        # x = self.X_train
        x = self.params['activation1'](self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.output(x)
        return x

    def convert_test_train_to_torch(self):
        self.X_train, self.X_test, self.y_train, self.y_test = torch.tensor(self.X_train.values), \
                                                               torch.tensor(self.X_test.values), \
                                                               torch.tensor(self.y_train.values).type(torch.LongTensor), \
                                                               torch.tensor(self.y_test.values).type(torch.LongTensor)

    def split_df_to_train_test(self, threshold_a, threshold_b=None, val=False):
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
        self.X = self.X.astype('float32')
        scaler = StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)
        self.X = pd.DataFrame(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)
        print(unique_pd(pd.Series(self.y_test)))
        self.convert_test_train_to_torch()
        if val == True:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.2,
                                                                                  random_state=42)

    def print_matrix(self, preds):
        df_corr = pd.DataFrame({'Y': self.y_test, 'YHat': preds})
        df_corr['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df_corr['Y'], df_corr['YHat'])]

        accuracy = df_corr['Correct'].sum() / len(df_corr)
        true_positive = pd.Series(df_corr['YHat'][df_corr['Correct'] == 1] == 0).sum()
        false_positive = pd.Series(df_corr['YHat'][df_corr['Correct'] == 0] == 0).sum()
        false_negative = pd.Series(df_corr['YHat'][df_corr['Correct'] == 0] == 1).sum()
        true_negative = pd.Series(df_corr['YHat'][df_corr['Correct'] == 1] == 1).sum()

        recall = true_positive / (true_positive + false_positive)
        precision = true_positive / (true_positive + false_positive)
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        print(f'recall is {"{:.2%}".format(recall)}\nprecision is {"{:.2%}".format(precision)}'
              f'\naccuracy is {"{:.2%}".format(accuracy)}\nSensitivity is {"{:.2%}".format(sensitivity)}'
              f'\nSpecificity is {"{:.2%}".format(specificity)}')

        print(
            f'accuracy is {"{:.2%}".format(accuracy)}\ntrue_positive is {true_positive}\nfalse_positive is {false_positive}\n'
            f'false_negative is {false_negative}\ntrue_negative is {true_negative}')
        return specificity

    def plot_loss(self, loss_arr, loss_arr_val):
        loss_arr_plot = [loss_arr[i].tolist() for i in range(0, 2000)]
        plt.plot(loss_arr_plot)
        plt.plot(loss_arr_val)
        plt.show()


if __name__ == "__main__":

    def objective(trial):

        params = {
            'learning_rate': trial.suggest_float('learning_rate', low=0.0001, high=0.01),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
            'activation1': trial.suggest_categorical('activation1', [F.leaky_relu, F.relu]),
            'activation2': trial.suggest_categorical('activation2', ['leaky_relu', 'relu']),
            'activation3': trial.suggest_categorical('activation3', ['leaky_relu', 'relu']),
            'activation4': trial.suggest_categorical('activation4', ['leaky_relu', 'relu']),
            'activation5': trial.suggest_categorical('activation5', ['leaky_relu', 'relu']),
            'l1': trial.suggest_int('l1', low=10, high=280, step=10),
            'l2': trial.suggest_int('l2', low=10, high=280, step=10),
            'l3': trial.suggest_int('l3', low=10, high=280, step=10),
            'l4': trial.suggest_int('l4', low=10, high=280, step=10),
            'l5': trial.suggest_int('l5', low=10, high=280, step=10),
            'dropout': trial.suggest_float('dropout', low=0.04, high=0.4, step=0.01),
        }

        model = Model(df, params)
        model.split_df_to_train_test(threshold_a=0.7, val=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model.params['learning_rate'])
        epochs = 50
        loss_arr = []
        loss_arr_val = []
        for i in range(epochs):
            y_pred = model.forward()
            loss = criterion(y_pred, model.y_train)
            loss_arr.append(loss)
            with torch.no_grad():
                y_pred_val = model.forward(model.X_val)
                loss_val = criterion(y_pred_val, model.y_val)
                loss_arr_val.append(loss_val)

            if i % 10 == 0:
                print(f'Epoch: {i} Loss: {loss}')
                print(f'Loss val: {loss_val}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = []
        with torch.no_grad():
            for val in model.X_test:
                y_pred = model.forward(val)
                preds.append(y_pred.argmax().item())

        # model.print_matrix(preds)
        # model.plot_loss(loss_arr, loss_arr_val)

        specificity = model.print_matrix(preds)

        return specificity


    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)

    '''[I 2022-12-08 16:03:59,615] Trial 36 finished with value: 0.5854765506807866 and parameters: {'learning_rate': 0.014434972975292901, 'optimizer': 'Adam', 'activation1': 'relu', 'activation2': 'relu', 'activation3': 'leaky_relu', 'activation4': 'relu', 'activation5': 'relu', 'l1': 230, 'l2': 120, 'l3': 90, 'l4': 120, 'l5': 230, 'dropout': 0.14}. Best is trial 30 with value: 0.6031653671593157.
'''
    '''[I 2022-12-08 16:12:00,316] Trial 38 finished with value: 0.5967648085651112 and parameters: {'learning_rate': 0.008932111800449834, 'optimizer': 'RMSprop', 'activation1': 'leaky_relu', 'activation2': 'relu', 'activation3': 'leaky_relu', 'activation4': 'relu', 'activation5': 'relu', 'l1': 100, 'l2': 180, 'l3': 200, 'l4': 40, 'l5': 270, 'dropout': 0.2}. Best is trial 30 with value: 0.6031653671593157.
r'''
    '''[I 2022-12-08 16:25:45,589] Trial 41 finished with value: 0.5990922844175491 and parameters: {'learning_rate': 0.008935430873648231, 'optimizer': 'RMSprop', 'activation1': 'leaky_relu', 'activation2': 'relu', 'activation3': 'leaky_relu', 'activation4': 'relu', 'activation5': 'relu', 'l1': 110, 'l2': 160, 'l3': 280, 'l4': 50, 'l5': 270, 'dropout': 0.18000000000000002}. Best is trial 30 with value: 0.6031653671593157.
r'''
    '''[I 2022-12-08 16:35:52,409] Trial 43 finished with value: 0.6494821366228325 and parameters: {'learning_rate': 0.004655344720768033, 'optimizer': 'RMSprop', 'activation1': 'leaky_relu', 'activation2': 'relu', 'activation3': 'leaky_relu', 'activation4': 'relu', 'activation5': 'relu', 'l1': 60, 'l2': 190, 'l3': 250, 'l4': 40, 'l5': 250, 'dropout': 0.18000000000000002}. Best is trial 43 with value: 0.6494821366228325.
r'''
    '''[I 2022-12-09 15:19:41,212] Trial 24 finished with value: 0.6066565809379728 and parameters: {'learning_rate': 0.004647040986556792, 'optimizer': 'SGD', 'activation1': 'leaky_relu', 'activation2': 'relu', 'activation3': 'relu', 'activation4': 'leaky_relu', 'activation5': 'relu', 'l1': 260, 'l2': 770, 'l3': 500, 'l4': 320, 'l5': 880, 'dropout': 0., 'epochs': 60}. Best is trial 24 with value: 0.6066565809379728.
recall is 84.74%
'''
