import numpy as np
import pandas as pd
import itertools

from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint

import matplotlib.pyplot as plt

def my_permutation_t_stat_1sample(sample, mean):
    t_stat = sum(map(lambda x: x - mean, sample))
    return  t_stat

def my_permutation_zero_distr_1sample(sample, mean, max_permutations = None):
    centered_sample = map(lambda x: x - mean, sample)
    if max_permutations:
        signs_array =set([tuple(x) for x in 2*np.random.randint(2,
                                                                size=(max_permutations,
                                                                      len(sample -1)))])
    else:
        signs_array = itertools.product([-1,1], repeat=len(sample))
    distr = [sum(centered_sample * np.array(signs)) for signs in signs_array]
    return distr

mouses_data = pd.read_csv('mirror_mouses.txt', header=None)
mouses_data.columns = ['proportion_of_time']

print(mouses_data)
print(mouses_data.describe())

plt.hist(mouses_data.proportion_of_time)
# plt.show()

interval = zconfint(mouses_data)
print('95% доверительный интервал для доли времени проведенной в комнате с зеркалом:')
print('[{}; {}]'.format(interval[0], interval[1]))

print('Проверка гипотезы по критерию знаков:')
print(sign_test(mouses_data, mu0=0.5))

m0 = 0.5
print('Проверка гипотезы по критерию знаковых рангов Вилкоксона:')
print(stats.wilcoxon(mouses_data.proportion_of_time - m0))

t_stat = my_permutation_t_stat_1sample(mouses_data.proportion_of_time, 0.5)
print('t_stat = ', t_stat)