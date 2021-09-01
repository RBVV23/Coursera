from sklearn import model_selection, linear_model, metrics, pipeline, preprocessing
import matplotlib.pyplot as plt
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
    ('feature_processing', pipeline.FeatureUnion(transformer_list=[
        ('binary_variable_processing', preprocessing.FunctionTransformer(lambda data: data.iloc[:, binary_data_indices])),
        ('numeric_variable_processing', pipeline.Pipeline(steps=[
            ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, numeric_data_indices])),
            ('scaling', preprocessing.StandardScaler(with_mean=0))
        ])),
        ('categorical_variable_processing', pipeline.Pipeline(steps=[
            ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, categorical_data_indices])),
            ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))
        ])),
    ])),
    ('model_fitting', regressor)
    ]
)

estimator.fit(train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, estimator.predict(test_data)))

print()
print('estimator.get_params().keys():')
print(estimator.get_params().keys())

parameters_grid = {
    'model_fitting__alpha' : [0.0001, 0.001, 0.1],
    'model_fitting__eta0' : [0.001, 0.05]
}

grid_cv = model_selection.GridSearchCV(estimator, parameters_grid, scoring='neg_mean_absolute_error', cv=4)
grid_cv.fit(train_data, train_labels)

print('grid_cv.best_score_ = ', grid_cv.best_score_)
print('grid_cv.best_params_ = ', grid_cv.best_params_)

test_predictions = grid_cv.best_estimator_.predict(test_data)
print(metrics.mean_absolute_error(test_labels, test_predictions))

print(test_labels[:20])
print(test_predictions[:20])

plt.figure(figsize=(16,6))
plt.grid(True)
plt.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color='red')
plt.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color='blue')
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.show()