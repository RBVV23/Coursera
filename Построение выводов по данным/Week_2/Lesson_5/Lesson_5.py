import numpy as np
import pandas as pd
import itertools

from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.weightstats import *
import matplotlib.pyplot as plt

def my_permutation_t_stat_ind(sample1, sample2):
    result = np.mean(sample1) - np.mean(sample2)
    return result

def my_get_random_combinations(n1, n2, max_combinations):
    index = range(n1 + n2)
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


seattle_data = pd.read_csv('seattle.txt', header=0, sep='\t')
print('seattle_data.shape = ', seattle_data.shape)
print(seattle_data.head())

price2001 = seattle_data[seattle_data.Year == 2001].Price
price2002 = seattle_data[seattle_data.Year == 2002].Price

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.grid()
plt.hist(price2001, color='r', label='2001')
plt.legend()
plt.xlabel('Стоимость недвижимости')
plt.subplot(1,2,2)
plt.grid()
plt.hist(price2002, color='b', label='2002')
plt.legend()
plt.xlabel('Стоимость недвижимости')
# plt.show()

print('Доверительный интервал (95%) для цен 2001 года:')
print(zconfint(price2001))
print('Доверительный интервал (95%) для цен 2002 года:')
print(zconfint(price2002))

print('Проверка гипотезы по критерию Манна-Уитни:')
print(stats.mannwhitneyu(price2001, price2002))

