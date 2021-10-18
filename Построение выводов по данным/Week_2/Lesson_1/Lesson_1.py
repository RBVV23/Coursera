import numpy as np
import pandas as pd

import scipy
from statsmodels.stats.weightstats import *
import matplotlib.pyplot as plt

data = pd.read_csv('ADHD.txt', sep=' ', header=0)
data.columns = ['Placebo', 'Methylphenidate']
data.plot.scatter('Placebo', 'Methylphenidate', c='r', s=30)
plt.grid()
plt.xlim((20,80))
plt.ylim((20,80))
plt.plot(range(100), c='black')
plt.show()

data.plot.hist()
plt.show()

result = stats.ttest_1samp(data.Placebo, 50.0)
print('stats.ttest_1samp(data.Placebo, 50.0):')
print(result)
interval = zconfint(data.Placebo)
print('95% доверительный интервал для среднего (группа плацебо): [{}; {}]'.format(interval[0], interval[1]))

# result = stats.ttest_1samp(data.Methylphenidate, 50.0)
# print('stats.ttest_1samp(data.Methylphenidate, 50.0):')
# print(result)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
stats.probplot(data.Placebo, dist='norm', plot=plt)
plt.subplot(2,2,2)
stats.probplot(data.Methylphenidate, dist='norm', plot=plt)
plt.show()

print('Критерий Шапиро-Уилка (группа плацебо): ')
print(stats.shapiro(data.Placebo))
print('Критерий Шапиро-Уилка (тестовая группа): ')
print(stats.shapiro(data.Methylphenidate))

print('Критерий Стьюдената для проверки равенства средних двух независимых выборок:')
print(stats.ttest_ind(data.Placebo, data.Methylphenidate, equal_var=False))
cm = CompareMeans(DescrStatsW(data.Methylphenidate), DescrStatsW(data.Placebo))
interval = cm.tconfint_diff(usevar='unequal')
print('95% доверительный интервал для разности: [{}; {}]'.format(interval[0], interval[1]))

stats.probplot(data.Methylphenidate - data.Placebo, dist='norm', plot=plt)
plt.show()

print('Критерий Шапиро-Уилка (для разности): ')
print(stats.shapiro(data.Methylphenidate - data.Placebo))


print('Критерий Стьюдената для проверки равенства средних двух связанных выборок:')
print(stats.ttest_rel(data.Placebo, data.Methylphenidate))

interval = DescrStatsW(data.Methylphenidate - data.Placebo).tconfint_mean()
print('95% доверительный интервал для разности: [{}; {}]'.format(interval[0], interval[1]))