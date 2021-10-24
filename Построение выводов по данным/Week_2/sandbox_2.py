import random

import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import samplesize_confint_proportion
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, datasets, linear_model, tree, ensemble
from statsmodels.stats.descriptivestats import sign_test
import itertools
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
    low = float(f - g)/n - z*sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    high = float(f - g)/n + z*sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    return low, high
def my_p_value(expect_mean=9.5, std=0.4, n=160, sample_mean=9.57, alpha=0.95, alternative='two-sided'):
    # z = stats.t.ppf((1+alpha)/2.0, n-1)
    z = (sample_mean - expect_mean)/(std/sqrt(n))
    # print('Z(Xn) = ', z)
    Fz = stats.t.ppf(0.05/2,n-1)
    S = 0.5*(1 + scipy.special.erf((z - 0)/sqrt(2*1**2)))
    # print('F(z) = ', abs(Fz))
    # print('S = ', S)
    if alternative == 'two-sided':
        p = 2*(1 - scipy.stats.norm.cdf(abs(z)))
    if alternative == 'less':
        p = scipy.stats.norm.cdf(z)
    if alternative == 'greater':
        p = 1 - scipy.stats.norm.cdf(z)
    # p = 2*(1 - stats.norm.cdf(abs(z)))
    # print('p = ', p)
    return p
def my_proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    p1 = sum(sample1) / n1
    n2 = len(sample2)
    p2 = sum(sample2) / n2
    P = (p1 * n1 + p2 * n2) / (n1 + n2)
    Z = (p1 - p2) / np.sqrt(P * (1 - P) * (1 / n1 + 1 / n2))
    return Z
def my_proportions_diff_z_test(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Недопустимое значения параметра "alternative"\n'
                         'допустимо: "two-sided", "less" или "greater"')

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(abs(z_stat)))
    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)
    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)
def my_proportions_diff_z_stat_rel(sample1, sample2):
    sample = list(zip(sample1, sample2))
    n = len(sample)
    f = 0
    g = 0
    for x in sample:
        if x[0] == 1 and x[1] == 0:
            f += 1
        if x[0] == 0 and x[1] == 1:
            g += 1
    result = (f-g)/np.sqrt(f+g - ((f-g)**2)/n)
    return result
def my_get_boostraps_samples(data, n_samples):
    L = len(data)
    indices = np.random.randint(0, L, (n_samples, L))
    samples = data[indices]
    return samples
def my_stat_intervals(stat, alpha=0.05):
    low, high = np.percentile(stat, [100*alpha/2., 100*(1 - alpha/2.)])
    return low, high
def my_permutation_test(sample1, sample2, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError('Недопустимое значения параметра "alternative"\n'
                         'допустимо: "two-sided", "less" или "greater"')
    t_stat = my_permutation_t_stat_ind(sample1, sample2)
    zero_distr = my_permutation_zero_dist_ind(sample1, sample2, max_permutations)

    if alternative == 'two-sided':
        res = sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr])/len(zero_distr)
    if alternative == 'less':
        res = sum([1. if x <= t_stat else 0. for x in zero_distr])/len(zero_distr)
    if alternative == 'greater':
        res = sum([1. if x >= t_stat else 0. for x in zero_distr])/len(zero_distr)
    return res
def my_get_random_combinations(n1, n2, max_combinations):
    index = list(range(n1 + n2))
    indices = set([tuple(index)])
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    result = [(index[:n1], index[n1:]) for index in indices]
    return result
def my_permutation_zero_dist_ind(sample1, sample2, max_combinations = None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n2 = len(sample2)
    n = len(joined_sample)

    if max_combinations:
        indices = my_get_random_combinations(n1, n2, max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) for index in itertools.combinations(range(n), n1)]
    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() for i in indices]
    return distr
def my_permutation_t_stat_ind(sample1, sample2):
    result = np.mean(sample1) - np.mean(sample2)
    return result

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

print('3.3. Достигаемый уровень значимости при альтернативе заразительности зевоты:')

n_test = 34
n_control = 16
test = np.array([1]*10 + [0]*(n_test-10))
control = np.array([1]*4 + [0]*(n_control-4))


z_stat = my_proportions_diff_z_stat_ind(test, control)
answer33 = round(my_proportions_diff_z_test(z_stat, alternative='greater'),4)
print('answer 3.3. = ', answer33)

df = pd.read_csv('banknotes.txt', header=0, sep='\t')
print(df.head())
data = df.drop('real', axis='columns')
# print(data.head())
target = df['real']

train_data, test_data, train_target, test_target = model_selection.train_test_split(data, target,
                                                                                    test_size=0.25,
                                                                                    random_state=1)
train_data_2 = train_data.drop(['X1', 'X2', 'X3'], axis='columns')
train_data_1 = train_data.drop(['X4', 'X5', 'X6'], axis='columns')
test_data_2 = test_data.drop(['X1', 'X2', 'X3'], axis='columns')
test_data_1 = test_data.drop(['X4', 'X5', 'X6'], axis='columns')

estimator_1 = linear_model.LogisticRegression(solver='liblinear')
estimator_1.fit(train_data_1, train_target)
predictions_1 = estimator_1.predict(test_data_1)
# print(test_target)
print('predictions_1:')
print(predictions_1)
accuracy_1 = metrics.accuracy_score(test_target, predictions_1)
print('Доля ошибок первого классификатора: ', 1-accuracy_1)
errors_1 = [0 if a == b else 1 for a,b in zip(predictions_1,test_target)]
print('errors_1')
print(errors_1)
estimator_2 = linear_model.LogisticRegression(solver='liblinear')
estimator_2.fit(train_data_2, train_target)
predictions_2 = estimator_2.predict(test_data_2)
print('predictions_2:')
print(predictions_2)
accuracy_2 = metrics.accuracy_score(test_target, predictions_2)
print('Доля ошибок второго классификатора: ', 1-accuracy_2)
errors_2 = [0 if a == b else 1 for a,b in zip(predictions_2,test_target)]
print('errors_2')
print(errors_2)

p_value = my_proportions_diff_z_test(my_proportions_diff_z_stat_rel(errors_1, errors_2))
print('3.4. Значение достижимого уровня значимости: ', p_value)

print('3.5. Доверительный интервал для разности долей ошибок двух классификаторов:')
print(my_proportions_confint_diff_rel(errors_1, errors_2))
print('3.6. Достигаемый уровень значимости для гипотезы о неэффективности программы: ')
control_mean = 525
control_std = 100
test_mean = 541.4
test_n = 100
answer36 = round(my_p_value(expect_mean=525, std=100, n=100, sample_mean=541.4, alpha=0.95, alternative='greater'),4)
print('answer 3.6. = ', answer36)

print('3.7. Достигаемый уровень значимости для гипотезы о неэффективности программы (с увеличенным средним): ')
control_mean = 525
control_std = 100
test_mean = 541.5
test_n = 100
answer37 = round(my_p_value(control_mean, control_std, test_n, test_mean, alpha=0.95, alternative='greater'),4)
print('answer 3.7. = ', answer37)

sample = np.array([49,58,75,110,112,132,151,276,281,362]) - 200

print('4.4. Достижимый уровень значимости для критерия знаковых рангов против двусторонней альтернативы:')
answer44 = round(stats.wilcoxon(sample, mode='approx')[1],4)
print('answer 4.4. = ', answer44)

print('4.5. Достижимый уровень значимости для критерия знаковых рангов против односторонней альтернативы:')
sample1 = np.array([22,22,15,13,19,19,18,20,21,13,13,15])
sample2 = np.array([17,18,18,15,12,4,14,15,10])
answer45 = round(stats.mannwhitneyu(sample1, sample2)[1],4)

print('answer 4.5. = ', answer45)

print('4.6. 95% доверительный интервал для разности средних температур воздуха при запусках при помощи бустрепа:')
df = pd.read_csv('challenger.txt', sep='\t', header=0)
df.columns=['Date', 'Temperature', 'Incident']
# print(df.head())
my_ones = df.Temperature[df.Incident == 1].values
my_zeros = df.Temperature[df.Incident == 0].values
np.random.seed(0)
new_ones = my_get_boostraps_samples(my_ones, 1000)
new_zeros = my_get_boostraps_samples(my_zeros, 1000)
zeros_mean = np.mean(new_zeros, axis=1)
ones_mean = np.mean(new_ones, axis=1)
my_list = list(map(lambda x: x[1] - x[0], zip(ones_mean, zeros_mean)))
# deltas = new_ones - new_zeros

print(my_stat_intervals(my_list))

answer46 = round(my_stat_intervals(my_list)[0],4)
print('answer 4.6. = ', answer46)

print('4.7. Проверка нулевой гипотезы при помощи перестоновочного критерия:')
np.random.seed = 0
answer47 = round(my_permutation_test(my_ones, my_zeros, max_permutations=10000),4)
print('answer 4.7. = ', answer47)