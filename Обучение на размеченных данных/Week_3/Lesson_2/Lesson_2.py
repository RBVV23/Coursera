from sklearn import model_selection, linear_model, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 100)

raw_data = pd.read_csv('bike_sharing_demand.csv', header=0, sep=',')
print(raw_data.head())
print(raw_data.shape)
print(raw_data.isnull().values.any())
print(raw_data.info())

# Описание признаков
# datetime - hourly date + timestamp
# season -
#     1 = spring, 2 = summer, 3 = fall, 4 = winter
# holiday - whether the day is considered a holiday
# workingday - whether the day is neither a weekend nor holiday
# weather -
#     1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentalsлО

raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)
raw_data['month'] = raw_data.datetime.apply(lambda x: x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x: x.hour)

print(raw_data.head())

train_data = raw_data.iloc[:-1000, :]
hold_out_test_data = raw_data.iloc[-1000:, :]
print('raw_data.shape = ', raw_data.shape)
print('train_data.shape = ', train_data.shape)
print('hold_out_test_data.shape = ', hold_out_test_data.shape)

print('\ntrain period from {} to {}'.format(train_data.datetime.min(), train_data.datetime.max()))
print('test period from {} to {}'.format(hold_out_test_data.datetime.min(), hold_out_test_data.datetime.max()))

train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count'], axis=1)

test_labels = hold_out_test_data['count'].values
test_data = hold_out_test_data.drop(['datetime', 'count'], axis=1)

plt.figure(figsize=(16,6))
plt.subplot(1, 2, 1)
plt.hist(train_labels)
plt.title('train data')
plt.subplot(1, 2, 2)
plt.hist(test_labels)
plt.title('test data')
plt.show()

numeric_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'month', 'hour']
train_data = train_data[numeric_columns]
test_data = test_data[numeric_columns]

print(train_data.head())
print(test_data.head())

regressor = linear_model.SGDRegressor(random_state=0, max_iter=5)
regressor.fit(train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, regressor.predict(test_data)))

print(test_labels[:10])
print(regressor.predict(test_data)[:10])
print('regressor.coef_ = ', regressor.coef_)

scaler = StandardScaler()
scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

regressor.fit(scaled_train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data)))

print(test_labels[:10])
print(regressor.predict(scaled_test_data)[:10])

print('regressor.coef_ = ', regressor.coef_)
print('regressor.coef_ = ', list(map(lambda x: round(x,2), regressor.coef_)))

print(train_data.head())
flag = np.all(train_data.registered + train_data.casual == train_labels)
print(flag)

train_data.drop(['registered', 'casual'], axis=1, inplace=True)
test_data.drop(['registered', 'casual'], axis=1, inplace=True)

scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

regressor.fit(scaled_train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, regressor.predict((scaled_test_data))))

print('regressor.coef_ = ', list(map(lambda x: round(x,2), regressor.coef_)))

pipeline = Pipeline(steps=[('scaling', scaler), ('regression', regressor)])
pipeline.fit(train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, pipeline.predict((test_data))))

print()
print("pipeline.get_params().keys()")
print(pipeline.get_params().keys())

parameters_grid = { 'regression__loss': ['huber', 'epsilon_insensitive', 'squared_loss'],
                    'regression__max_iter': [3, 5, 10, 50],
                    'regression__penalty': ['l1', 'l2', 'none'],
                    'regression__alpha': [0.0001, 0.01],
                    'scaling__with_mean': [0., 0.5]}

grid_cv = model_selection.GridSearchCV(pipeline, parameters_grid,
                                       scoring='neg_mean_absolute_error', cv=4)
grid_cv.fit(train_data, train_labels)

print('grid_cv.best_score_ = ', grid_cv.best_score_)
print('grid_cv.best_params_ = ', grid_cv.best_params_)

print(metrics.mean_absolute_error(test_labels, grid_cv.best_estimator_.predict(test_data)))
print('np.mean(test_labels)', np.mean(test_labels))

test_predictions = grid_cv.best_estimator_.predict(test_data)
print(test_labels[:10])
print(test_predictions[:10])

plt.figure(figsize=(16,6))
plt.subplot(1, 2, 1)
plt.grid(True)
plt.scatter(train_labels, pipeline.predict(train_data), alpha=0.5, color='red')
plt.scatter(test_labels, pipeline.predict(test_data), alpha=0.5, color='blue')
plt.title('no parameters setting')
plt.xlim(-100,1100)
plt.ylim(-100,1100)

plt.subplot(1, 2, 2)
plt.grid(True)
plt.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color='red')
plt.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color='blue')
plt.title('grid search')
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.show()