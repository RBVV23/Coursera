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

print(my_permutation_t_stat_1sample(mouses_data.proportion0.5))