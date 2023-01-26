import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from func import *
from classification_model import ClassificationModel
from NN_pytorch import Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

path = r'Data/gdf.csv'
df = pd.read_csv(path, low_memory=False)


# df = df_himalayan

df = df[df['FLUORIDE'] < 30]  # remove the outliers
df_get_dummies = pd.get_dummies(df.loc[:, ['SITE_TYPE', 'STATE_NAME']], columns=['SITE_TYPE', 'STATE_NAME'])
df_get_dummies = df_get_dummies.copy()
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

dict_of_divided = {'dakan': country_list_of_dakan, 'himalayan': country_list_of_himalayan,
                   'lowland': country_list_of_lowland}

clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)




# for name_countries in dict_of_divided:
#     for i in tqdm(range(len(df['STATE_NAME']))):
#         if i in dict_of_divided[name_countries]:
#             df[dict_of_divided].loc[i] = 1
#         else:
#             df[dict_of_divided].loc[i] = 0


def add_dummy_columns(df, data, column_name):
    # Create a new dataframe with all values set to 0
    dummy_df = pd.DataFrame(0, index=df.index, columns=data.keys())

    # Iterate through the rows of the dataframe
    for index, row in df.iterrows():
        # Get the value of the specified column for the current row
        value = row[column_name]

        # Iterate through the keys in the dictionary
        for key in data.keys():
            # Get the list of values for the current key
            values = data[key]

            # Check if the value is in the list
            if value in values:
                # If the value is in the list, set the value of the corresponding column in the dummy dataframe to 1
                dummy_df.at[index, key] = 1

    # Join the dummy dataframe with the original dataframe
    df = df.join(dummy_df)

    # Return the modified dataframe
    return df

def convert(df):
    X = df.loc[:, 'AET.tif':].copy()
    X = X.drop(['FLUORIDE'], axis=1).fillna(0)
    return X


def predict_sample(sample, df, clf_model, nn_model, first_run=True):

    if first_run:
        clf_model = ClassificationModel(df)
        nn_model = Model(df)
        clf_model.split_df_to_train_test(0.7)
        clf_model.fit(clf_GradientBoosting)
        nn_model.split_df_to_train_test(threshold_a=0.7, val=True)
        nn_model.fit(lr=0.004647, epochs=60)
        nn_pred = nn_model.predict(nn_model.X_test)
    sample = convert(sample)
    clf_pred = clf_model.predict(sample)
    sample = torch.tensor(sample.values).type('torch.FloatTensor')
    nn_pred = nn_model.predict(sample)
    return [clf_pred, nn_pred], clf_model, nn_model







df_sample = df_dakan.reset_index()
first_run = True
_, clf_model, nn_model = predict_sample(df.loc[1:1, :], df, ClassificationModel, Model, first_run)
sample = df_sample.loc[1:1, :]
_, clf_model_dakan, nn_model_dakan = predict_sample(sample, df_dakan, ClassificationModel, Model, first_run)
_, clf_model_himalayan, nn_model_himalayan = predict_sample(sample, df_himalayan, ClassificationModel, Model, first_run)
_, clf_model_lowland, nn_model_lowland = predict_sample(sample, df_lowland, ClassificationModel, Model, first_run)




#functuon that recive df  and run all over the df whit loop, and return predict list
_, X_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
X_test_n = add_dummy_columns(X_test, dict_of_divided, 'STATE_NAME')
bacic = 0
dakan = 0
himalayan = 0
lowland = 0
other = 0
first_run = False
predictions = []
for i in tqdm(range(len(X_test))):
    sample = X_test.loc[i:i, :]
    pred, _, _ = predict_sample(sample, df, clf_model, nn_model, first_run)

    if pred[0] == pred[1]:
        predictions.append(pred)
        bacic += 1
    elif X_test_n.loc[i:i, 'dakan'].any() == 1:
        dakan += 1
        pred, _, _ = predict_sample(sample, df_dakan, clf_model_dakan, nn_model_dakan, first_run)
        predictions.append(pred)
    elif (X_test_n.loc[i:i, 'himalayan'] == 1).any():
        himalayan += 1

        pred, _, _ = predict_sample(sample, df_himalayan, clf_model_himalayan, nn_model_himalayan, first_run)
        predictions.append(pred)
    elif (X_test_n.loc[i:i, 'lowland'] == 1).any():
        lowland += 1
        pred, _, _ = predict_sample(sample, df_lowland, clf_model_lowland, nn_model_lowland, first_run)
        predictions.append(pred)
    else:
        other += 1
        predictions.append(pred)

a = np.array(predictions)
(a[:, :1] == a[:, 1:2]).sum()





# for i, j in tqdm(enumerate(df['STATE_NAME'])):
#     if i in country_list_of_dakan:
#         df['dakan'].loc[j] = 1
#     else:
#         df['dakan'].loc[j] = 0
