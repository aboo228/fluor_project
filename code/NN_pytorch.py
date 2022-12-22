import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from func import unique_pd, find_and_replace_not_num_values, isfloat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

df.copy().to_csv('df for colab.csv', index=False)


class Model(nn.Module):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.num_fetchers = len(df.columns)
        self.lr, self.epochs = None, None
        self.fc1 = nn.Linear(in_features=self.num_fetchers, out_features=260)
        self.fc2 = nn.Linear(in_features=260, out_features=770)
        self.dropout1 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(in_features=770, out_features=500)
        self.fc4 = nn.Linear(in_features=500, out_features=320)
        self.fc5 = nn.Linear(in_features=320, out_features=880)
        self.output = nn.Linear(in_features=880, out_features=2)

    def forward(self, x=None):
        if x is None:
            x = self.X_train
        # x = self.X_train
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.output(x)
        return x

    def convert_test_train_to_torch(self):
        self.X_train, self.X_test, self.y_train, self.y_test = torch.tensor(self.X_train.values), \
                                                               torch.tensor(self.X_test.values), \
                                                               torch.tensor(self.y_train.values).type(torch.LongTensor), \
                                                               torch.tensor(self.y_test.values).type(torch.LongTensor)
    def split_df_to_train_test(self, threshold_a, threshold_b=None, val= False):
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


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print(unique_pd(pd.Series(self.y_test)))
        self.convert_test_train_to_torch()
        if val == True:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2,
                                                                                    random_state=42)

    def print_matrix(self, y):
        self.y_test = y
        df_corr = pd.DataFrame({'Y': self.y_test, 'YPred': self.preds})
        df_corr['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df_corr['Y'], df_corr['YPred'])]

        accuracy = df_corr['Correct'].sum() / len(df_corr)
        true_positive = pd.Series(df_corr['YPred'][df_corr['Correct'] == 1] == 0).sum()
        false_positive = pd.Series(df_corr['YPred'][df_corr['Correct'] == 0] == 0).sum()
        false_negative = pd.Series(df_corr['YPred'][df_corr['Correct'] == 0] == 1).sum()
        true_negative = pd.Series(df_corr['YPred'][df_corr['Correct'] == 1] == 1).sum()

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

        matrix = {'recall': recall, 'precision': precision,
                  'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity}
        return matrix

    def plot_loss(self):
        self.loss_arr_plot = [self.loss_arr[i].tolist() for i in range(0, self.epochs)]
        plt.plot(self.loss_arr_plot)
        plt.plot(self.loss_arr_val)
        plt.show()

    def fit(self, lr=0.005, epochs=50):
        self.lr = lr
        self.epochs = epochs
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.loss_arr = []
        self.loss_arr_val = []
        for i in range(self.epochs):
            y_pred = self.forward()
            loss = criterion(y_pred, self.y_train)
            self.loss_arr.append(loss)
            with torch.no_grad():
                y_pred_val = self.forward(self.X_val)
                loss_val = criterion(y_pred_val, self.y_val)
                self.loss_arr_val.append(loss_val)


            if i % 10 == 0:
                print(f'Epoch: {i} Loss: {loss}')
                print(f'Loss val: {loss_val}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    def predict(self, X):
        self.preds = []
        with torch.no_grad():
           for x in X:
               y_pred = self.forward(x)
               self.preds.append(y_pred.argmax().item())
        if len(self.preds) == 1:
            print(self.preds)
            return int(self.preds)


if __name__ == "__main__":

    model = Model(df)
    model.split_df_to_train_test(threshold_a=0.7, val=True)
    model.fit(lr=0.004647, epochs=5)
    model.predict(model.X_test)
    model.print_matrix(model.y_test)
    model.plot_loss()














