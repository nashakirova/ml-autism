from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from prepare import dataset
import pandas as pd


ready_dataset = dataset.copy()

scaler = StandardScaler()
num_cols = ['age', 'result']
ready_dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

ready_dataset.head()

encoder = OneHotEncoder(sparse=False)
cat_cols = ['gender', 'ethnicity', 'jundice', 'country_of_res']

# Encode Categorical Data
dataset_encoded = pd.DataFrame(encoder.fit_transform(ready_dataset[cat_cols]))
dataset_encoded.columns = encoder.get_feature_names_out(cat_cols)

# Replace Categotical Data with Encoded Data
ready_dataset = ready_dataset.drop(cat_cols ,axis=1)
ready_dataset = pd.concat([dataset_encoded, ready_dataset], axis=1)

# Encode target value
ready_dataset=ready_dataset.replace({'no': 0, 'yes': 1, 'NO': 0, 'YES': 1})

print('Shape of dataframe:', ready_dataset.shape)
ready_dataset.head()