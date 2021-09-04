import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def calculate_means(numeric_data):
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
#       indices = np.nonzero(~numeric_data.iloc[:,j].isnull())[0] - оригинальная строка (не работает)
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull().values)[0] # - авторская версия
        correction = np.amax(to_sum[indices])
#         to_sum /= correction - первая строка из ненужной пары
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
#         means[j] *= correction - вторая строка из ненужной пары
    return pd.Series(means, numeric_data.columns)

data = pd.read_csv('data.csv')
print('data.shape = ', data.shape)

X = data.drop('Grant.Status', 1)
y = data['Grant.Status']

print('data.head()')
data.head()

numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3',
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))

data.dropna().shape

means = data[numeric_cols].mean()
print('means = ', means)

numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3',
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']

means = data[numeric_cols].mean()

X = data.drop('Grant.Status', 1)
# y = data['Grant.Status']
X_real_mean = X[numeric_cols]
# X_real_zeros = X

for col in numeric_cols:
    indices = X[col].isnull()
#     print(X[col])
#     print(indices)
    X_real_mean[col][indices] = X_real_mean[col][indices].apply(lambda x: means[col])
#     print(X_real_mean[col])


X = data.drop('Grant.Status', 1)
X_real_zeros = X[numeric_cols]
for col in numeric_cols:
    indices = X[col].isnull()
#     print(X[col])
#     print(indices)
    X_real_zeros[col][indices] = X_real_zeros[col][indices].apply(lambda x: 0)
#     print(X_real_zeros[col])

X = data.drop('Grant.Status', 1)
X_cat = X[categorical_cols]
for col in categorical_cols:
    indices = X[col].isnull()
#     print(X[col])
#     print(indices)
    X_cat[col][indices] = X_cat[col][indices].apply(lambda x: 'NA')
    X_cat[col] = X_cat[col].apply(lambda x: str(x))
#     print(X_real_zeros[col])
# print(X_cat)
X = data.drop('Grant.Status', 1)
print('X_cat', X_cat)

categorial_data = pd.DataFrame({'sex': ['male', 'female', 'male', 'female'],
                                'nationality': ['American', 'European', 'Asian', 'European']})
print('Исходные данные:\n')
print(categorial_data)
encoder = DV(sparse = False)
encoded_data = encoder.fit_transform(categorial_data.T.to_dict().values())
print('\nЗакодированные данные:\n')
print(encoded_data)

encoder = DV(sparse = False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())

(X_train_real_zeros,
 X_test_real_zeros,
 y_train, y_test) = train_test_split(X_real_zeros, y,
                                     test_size=0.3,
                                     random_state=0)
(X_train_real_mean,
 X_test_real_mean) = train_test_split(X_real_mean,
                                      test_size=0.3,
                                      random_state=0)
(X_train_cat_oh,
 X_test_cat_oh) = train_test_split(X_cat_oh,
                                   test_size=0.3,
                                   random_state=0)