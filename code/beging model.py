import pandas as pd
from tqdm import tqdm
from func import *
from classification_model import ClassificationModel
from NN_pytorch import Model
from sklearn.ensemble import GradientBoostingClassifier

path = r'Data/gdf.csv'
df = pd.read_csv(path, low_memory=False)

country_list_of_dakan = ['Andhra Pradesh', 'Dadra And Nagar Haveli', 'Goa', 'Karnataka',
                         'Kerala', 'Maharashtra', 'Odisha', 'Pondicherry', 'Tamil Nadu', 'Telangana', ]
country_list_of_himalayan = ['Arunachal Pradesh', 'Assam', 'Himachal Pradesh', 'Jammu And Kashmir',
                             'Meghalaya', 'Nagaland', 'Tripura', 'Uttarakhand', ]
country_list_of_lowland = ['Bihar', 'Chhattisgarh', 'Delhi', 'Gujarat', 'Haryana', 'Jharkhand',
                           'Punjab', 'Rajasthan', 'Uttar Pradesh', 'West Bengal']

df_dakan = df.query(f'STATE_NAME == {country_list_of_dakan}')
df_himalayan = df.query(f'STATE_NAME == {country_list_of_himalayan}')
df_lowland = df.query(f'STATE_NAME == {country_list_of_lowland}')

# df = df_himalayan
df_get_dummies = pd.get_dummies(df.loc[:, ['SITE_TYPE', 'STATE_NAME']], columns=['SITE_TYPE', 'STATE_NAME'])
df_get_dummies = df_get_dummies.copy()
df = pd.concat([df, df_get_dummies], axis=1)

clf_GradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

dict_of_divided = {'dakan': country_list_of_dakan, 'himalayan': country_list_of_himalayan,
                   'lowland': country_list_of_lowland}


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


n_df = add_dummy_columns(df, dict_of_divided, 'STATE_NAME')


def fit_beging_model():

# self = ClassificationModel(df)
# self.split_df_to_train_test(0.7)
# self.fit(clf_GradientBoosting)
# self.predict(self.X_test)
# matrix = self.confusion_matrix(self.y_test)
# self.confusion_df('STATE_NAME', to_print=True)
#
# model = Model(df)
# model.split_df_to_train_test(threshold_a=0.7, val=True)
# model.fit(lr=0.004647, epochs=5)
# model.predict(model.X_test)
# model.print_matrix(model.y_test)
# model.plot_loss()
#
# for i, j in tqdm(enumerate(df['STATE_NAME'])):
#     if i in country_list_of_dakan:
#         df['dakan'].loc[j] = 1
#     else:
#         df['dakan'].loc[j] = 0
