# Import libraries
## Basic libs
import pandas as pd
import numpy as np
import warnings
## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Configure libraries
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('seaborn')
# Load dataset
dataset = pd.read_csv('./autism_screening.csv')
print('Shape of dataframe:', dataset.shape)
dataset.head()

print('Class distribution: ', dataset['Class/ASD'].value_counts())
## We see that the data is not distributed evenly, ration is closer to 3:1 than 1:1

print('Validate that the data is not missing')
print(dataset.isnull().sum())
## We see that in two rows age is missing. We don't want to remove the valueable column entirely,
# hence let's implant the data
column = dataset['age']
column.fillna(column.mean(), inplace=True)
print('Validate that data no longer has empty values')
print(dataset.isnull().sum())

## Visually inspecting data, I see also that sometimes ethnicity is not set with a proper value and ? is used.
# I still would prefer not to eliminate the data as it provides a flavor in research, especially given that
# country is always populated
# print(dataset[dataset['ethnicity'] == '?'].value_counts())
# print(dataset[dataset['country_of_res'] == '?'].value_counts())
## As age_desc is validated to be 18 and more all the time with our implanted 2 rows of data, we can safely
# delete this column
# print(dataset[dataset['age_desc'] != '18 and more'].value_counts())
dataset.drop(['age_desc'], axis=1)
