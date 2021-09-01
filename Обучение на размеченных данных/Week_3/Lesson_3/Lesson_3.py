from sklearn import model_selection, linear_model, metrics, pipeline, preprocessing

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 100)

raw_data = pd.read_csv('bike_sharing_demand.csv', header=0, sep=',')

# print(raw_data.head())

raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)
raw_data['month'] = raw_data.datetime.apply(lambda x: x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x: x.hour)

# print(raw_data.head())

train_data = raw_data.iloc[:-1000, :]
test_data = raw_data.iloc[-1000:, :]

print('raw_data.shape = ', raw_data.shape)
print('train_data.shape = ', train_data.shape)
print('test_data.shape = ', test_data.shape)

train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count', 'registered', 'casual'], axis = 1)
test_labels = test_data['count'].values
test_data = test_data.drop(['datetime', 'count', 'registered', 'casual'], axis = 1)

print(train_data.head())

binary_data_columns = ['holiday', 'workingday']
binary_data_indices = np.array([column in binary_data_columns for column in train_data.columns],
                               dtype=bool)
print(binary_data_indices)

categorical_data_columns = ['season',  'weather', 'month']
categorical_data_indices = np.array([column in categorical_data_columns for column in train_data.columns],
                                    dtype=bool)
print(categorical_data_indices)

numeric_data_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
numeric_data_indices = np.array([column in numeric_data_columns for column in train_data.columns],
                                    dtype=bool)
print(numeric_data_indices)

# print(numeric_data_indices + categorical_data_indices + binary_data_indices)

regressor = linear_model.SGDRegressor(random_state=0, max_iter=3, loss='squared_loss', penalty='l2')

estimator = pipeline.Pipeline(steps=[
    ('feature_processing')
])