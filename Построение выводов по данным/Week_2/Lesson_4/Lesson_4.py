import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint


def my_permutation_t_stat_1sample(sample, mean):
    t_stat = sum(list(map(lambda x: x - mean, sample)))
    return  t_stat
def my_permutation_zero_distr_1sample(sample, mean, max_permutations = None):
    centered_sample = list(map(lambda x: x - mean, sample))
    if max_permutations:
        signs_array = set([tuple(x) for x in 2*np.random.randint(2,
                                                                size=(max_permutations,
                                                                      len(sample))) - 1])
    else:
        signs_array = itertools.product([-1,1], repeat=len(sample))
    distr = [sum(centered_sample * np.array(signs)) for signs in signs_array]
    return distr
def my_permutation_test(sample, mean, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError('Недопустимое значения параметра "alternative"\n'
                         'допустимо: "two-sided", "less" или "greater"')
    t_stat = my_permutation_t_stat_1sample(sample, mean)
    zero_distr = my_permutation_zero_distr_1sample(sample, mean, max_permutations)

    if alternative == 'two-sided':
        res = sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr])/len(zero_distr)
    if alternative == 'less':
        res = sum([1. if x <= t_stat else 0. for x in zero_distr])/len(zero_distr)
    if alternative == 'greater':
        res = sum([1. if x >= t_stat else 0. for x in zero_distr])/len(zero_distr)
    return res


weight_data = pd.read_csv('weight.txt', header=0, sep='\t')
print(weight_data.head())
print(weight_data.describe)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.hist(weight_data.Before, color='r', label='До терапии')
plt.xlabel('Масса тела (фунты)')
plt.legend()
plt.grid()
plt.subplot(1,2,2)
plt.hist(weight_data.After, color='g', label='После терапии')
plt.xlabel('Масса тела (фунты)')
plt.legend()
plt.grid()
plt.show()

print('Доверительный интервал для срерднего значения массы ДО терапии:', zconfint(weight_data.Before))
print('Доверительный интервал для срерднего значения массы ПОСЛЕ терапии:', zconfint(weight_data.After))

plt.figure(figsize=(12,4))
plt.hist(weight_data.After - weight_data.Before, color='b', label='После терапии')
plt.xlabel('Прирост масса тела (фунты)')
plt.legend()
plt.grid()
plt.show()

print('Проверка гипотезы по критерию знаков:')
print(sign_test(weight_data.After - weight_data.Before, mu0=0))

print('Проверка гипотезы по критерию Вилкоксона (для разности выборок):')
print(stats.wilcoxon(weight_data.After - weight_data.Before))
print('Проверка гипотезы по критерию Вилкоксона (для связанных выборок):')
print(stats.wilcoxon(weight_data.After, weight_data.Before))

plt.figure(figsize=(12,4))
plt.hist(my_permutation_zero_distr_1sample(weight_data.After - weight_data.Before, mean=0, max_permutations=10000))
plt.grid()
plt.show()

print('Проверка гипотезы при помощи перестановочного критерия (число перестановок - 1000)')
print('p_value = ', my_permutation_test(weight_data.After - weight_data.Before, mean=0, max_permutations=1000))
print('Проверка гипотезы при помощи перестановочного критерия (число перестановок - 50000)')
print('p_value = ', my_permutation_test(weight_data.After - weight_data.Before, mean=0, max_permutations=50000))