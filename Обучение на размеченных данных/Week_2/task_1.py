from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

data = pd.read_csv('bikes_rent.csv')
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 15)
print(data.head())
print(data.info())

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))
for idx, feature in enumerate(data.columns[:-1]):
    # print([idx // 4, idx % 4])
    data.plot(feature, "cnt", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
# plt.show()

print('\n======================== МАТРИЦА КОРРЕЛЯЦИЙ ПИРСОНА ========================')
print(data.corr())
print('\t ========================================================================================================\n')

str_task = 'temp, atemp, hum, windspeed(mph), windspeed(ms) и cnt'
# my_str = re.split('[^a-z]', str_task)
# while True:
#     try:
#         my_str.remove('')
#     except ValueError:
#         break
# print(my_str)


print('\n======================== СРЕДНИЕ ЗНАЧЕНИЯ ПО ПРИЗНАКАМ ========================')
print(np.mean(data, axis=0))
print('\t ========================================================================================================\n')

data_shuffled = shuffle(data, random_state=123)
X = scale(data_shuffled[data_shuffled.columns[:-1]])
y = data_shuffled["cnt"]

regressor = LinearRegression()
regressor.fit(X, y)
for col, cof in zip(data.columns, regressor.coef_):
    print('{}: {:.3f}'.format(col, cof))