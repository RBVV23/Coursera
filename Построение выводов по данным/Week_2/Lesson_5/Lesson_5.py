import itertools
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import *


def my_permutation_t_stat_ind(sample1, sample2):
    result = np.mean(sample1) - np.mean(sample2)
    return result
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
plt.show()

print('Доверительный интервал (95%) для цен 2001 года:')
print(zconfint(price2001))
print('Доверительный интервал (95%) для цен 2002 года:')
print(zconfint(price2002))

print('Проверка гипотезы по критерию Манна-Уитни:')
print(stats.mannwhitneyu(price2001, price2002))

plt.figure(figsize=(12,4))
plt.hist(my_permutation_zero_dist_ind(price2001, price2002, max_combinations=1000))
plt.show()

print('Проверка гипотезы при помощи перестановочного критерия (число перестановок - 10000)')
print('p_value = ', my_permutation_test(price2001, price2002, max_permutations=10000))
print('Проверка гипотезы при помощи перестановочного критерия (число перестановок - 50000)')
print('p_value = ', my_permutation_test(price2001, price2002, max_permutations=50000))