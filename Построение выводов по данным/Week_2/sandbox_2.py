import pandas as pd
# print(pd.__version__)
import numpy as np
# print(np.__version__)
from math import sqrt
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import samplesize_confint_proportion
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, datasets, linear_model, tree, ensemble
import scipy

def my_p_value(expect_mean=9.5, std=0.4, n=160, sample_mean=9.57, alpha=0.95):
    # z = stats.t.ppf((1+alpha)/2.0, n-1)
    z = (sample_mean - expect_mean)/(std/sqrt(n))
    # print('Z(Xn) = ', z)
    Fz = stats.t.ppf(0.05/2,n-1)
    S = 0.5*(1 + scipy.special.erf((z - 0)/sqrt(2*1**2)))
    # print('F(z) = ', abs(Fz))
    # print('S = ', S)
    p = 2*(1 - stats.norm.cdf(abs(z)))
    # print('p = ', p)
    return p

print('2.4. Достигаемый уровень значимости для гипотезы, что среднее значение уровня кальция отличается от среднего:')
answer24 = round(my_p_value(expect_mean=9.5, std=0.4, n=160, sample_mean=9.57, alpha=0.95),4)
print('answer 2.4. = ' ,answer24)

df = pd.read_csv('diamonds.txt', sep='\t', header=0)
print(df.head())
data = df.drop('price', axis='columns')
# print(data.head())
target = df.price
# print(target.head())

train_data, test_data, train_target, test_target = model_selection.train_test_split(data, target,
                                                                                    test_size=0.25,
                                                                                    random_state=1)
estimator_1 = linear_model.LinearRegression()
estimator_1.fit(train_data, train_target)
predictions_1 = estimator_1.predict(test_data)
# print(predictions)
error_1 = metrics.mean_absolute_error(predictions_1, test_target)
print('Ошибка логистической регрессии: ', error_1)

estimator_2 = ensemble.RandomForestRegressor(n_estimators=10, random_state=1)
estimator_2.fit(train_data, train_target)
predictions_2 = estimator_2.predict(test_data)
# print(predictions)
error_2 = metrics.mean_absolute_error(predictions_2, test_target)
print('Ошибка случайного леса: ', error_2)