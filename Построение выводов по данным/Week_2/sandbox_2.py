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
from statsmodels.stats.weightstats import *

def my_proportions_confint_diff_rel(sample1, sample2, alpha = 0.05):
    z = stats.norm.ppf(1 - alpha/2.)
    sample = list(zip(sample1, sample2))
    n = len(sample)
    f = 0
    g = 0
    for x in sample:
        if x[0] == 1 and x[1] == 0:
            f += 1
        if x[0] == 0 and x[1] == 1:
            g += 1
    # print('f = ',f)
    # print('g = ',g)
    low = float(f - g)/n - z*sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    high = float(f - g)/n + z*sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    return low, high

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
# print(predictions_1)
error_1 = metrics.mean_absolute_error(predictions_1, test_target)
print('Средняя абсолютная ошибка логистической регрессии: ', error_1)
errors_1 = abs(test_target - predictions_1)
# print(errors_1)
std_1 = errors_1.std(ddof=1)

estimator_2 = ensemble.RandomForestRegressor(n_estimators=10, random_state=1)
estimator_2.fit(train_data, train_target)
predictions_2 = estimator_2.predict(test_data)
# print(predictions_2)
error_2 = metrics.mean_absolute_error(predictions_2, test_target)
print('Средняя абсолютная ошибка случайного леса: ', error_2)
errors_2 = abs(test_target - predictions_2)
# print(errors_2)
std_2 = errors_2.std(ddof=1)

print('Критерий Стьюдената для проверки равенства средних двух связанных выборок:')
print(stats.ttest_rel(errors_1, errors_2))
print(stats.ttest_rel(abs(test_target - estimator_1.predict(test_data)),
                abs(test_target - estimator_2.predict(test_data))))

print('Доверительный интервал для разности средних значений зависмых выборок:')
interval = DescrStatsW(errors_1 - errors_2).tconfint_mean()
print('[{}; {}]'.format(interval[0], interval[1]))