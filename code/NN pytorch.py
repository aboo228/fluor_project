import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from func import unique_pd, find_and_replace_not_num_values, isfloat
from sklearn.model_selection import train_test_split


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


class ANN(nn.Module):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.fc1 = nn.Linear(in_features=93, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=2)

    def forward(self, x=None):
        if x is None:
            x = self.X_train
        # x = self.X_train
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x

    def convert_test_train_to_torch(self):
        self.X_train, self.X_test, self.y_train, self.y_test = torch.tensor(self.X_train.values), \
                                                               torch.tensor(self.X_test.values), \
                                                               torch.tensor(self.y_train.values).type(torch.LongTensor), \
                                                               torch.tensor(self.y_test.values).type(torch.LongTensor)
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
        self.X = self.X.astype('float32')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print(unique_pd(pd.Series(self.y_test)))
        self.convert_test_train_to_torch()





model = ANN(df)
model.split_df_to_train_test(threshold_a=0.7)
# model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# % % time
epochs = 100
loss_arr = []
for i in range(epochs):
    y_hat = model.forward()
    loss = criterion(y_hat, model.y_train)
    loss_arr.append(loss)

    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

preds = []
with torch.no_grad():
   for val in model.X_test:
       y_hat = model.forward(val)
       preds.append(y_hat.argmax().item())

df_corr = pd.DataFrame({'Y': model.y_test, 'YHat': preds})
df_corr['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df_corr['Y'], df_corr['YHat'])]

accuracy = df_corr['Correct'].sum()/len(df_corr)
true_positive = df_corr[df_corr['Y'] == 0]['Correct'].sum()
false_negative = df_corr['YHat'][df_corr['YHat'] == 1].sum()
true_negative = df_corr[df_corr['Y'] == 1]['Correct'].sum()
false_positive = df_corr['YHat'][df_corr['YHat'] == 0].sum()

print(f'accuracy is {"{:.2%}".format(accuracy)}\ntrue_positive is {true_positive}\nfalse_negative is {false_negative}\n'
      f'true_negative is {true_negative}\nfalse_positive is {false_positive}')

df_corr[df_corr['Y'] == 1]['Correct'].sum()
# df_corr.loc[df_corr['Y'] == 1, df_corr['Correct'].sum()]