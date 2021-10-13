import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
from math import sqrt
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import samplesize_confint_proportion
import matplotlib.pyplot as plt


def my_interval(X, alpha=0.95, precision=4, norm=True, flag=False):
    n = len(X)
    sample_mean=X.mean()
    sample_std = X.std(ddof=1)
    z = stats.t.ppf((1+alpha)/2.0, n-1)
    print('z (t) = ',z)
    if flag:
        print('sample_mean = ', sample_mean)
        print('sample_std = ', sample_std)

    low = sample_mean - z*sample_std/sqrt(n)
    low = round(low, precision)

    high = sample_mean + z*sample_std/sqrt(n)
    high = round(high, precision)
    print('Доверительный интервал {}%: ({} - {})'.format(100*alpha, low, high))
    print()
    return low, high

# A = np.array([3,2,3,4,3])
# my_interval(A)


water = pd.read_csv('water.txt', sep='\t', header=0)
# print(water.head())

my_X = water['mortality']

print('1.2. Постройте 95% доверительный интервал для средней годовой смертности в больших городах:')
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
my_interval(my_X, precision=4)

print('1.3. Постройте 95% доверительный интервал для средней годовой смертности по всем южным городам:')
df = water[water['location'] == 'South']
my_X = df['mortality']
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
my_interval(df['mortality'], precision=4)

print('1.4. Постройте 95% доверительный интервал для средней годовой смертности по всем северным городам')
df = water[water['location'] == 'North']
my_X = df['mortality']
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
my_interval(df['mortality'], precision=4)

print('1.5. 95% доверительные интервалы для средней жёсткости воды в северных и южных городах:')

print(_tconfint_generic(water[water.location == 'North'].hardness.mean(), water[water.location == 'North'].hardness.std(ddof=1) / np.sqrt(len(water[water.location == 'North'])), len(water[water.location == 'North']) - 1, 0.05, 'two-sided'))
df = water[water['location'] == 'North']
my_interval(df['hardness'], precision=4)

print(_tconfint_generic(water[water.location == 'South'].hardness.mean(), water[water.location == 'South'].hardness.std(ddof=1) / np.sqrt(len(water[water.location == 'South'])), len(water[water.location == 'South']) - 1, 0.05, 'two-sided'))
df = water[water['location'] == 'South']
my_interval(df['hardness'], precision=4)

print('2.2. Нормальный доверительный интервал для доли носителей варианта 13910T в популяции майя:')

sample_mean = 1/50
n = 50
sample_std = sqrt((49*(1/50)**2 + (49/50)**2)/(n-1))
z = stats.norm.ppf(0.975)
low = sample_mean - z*sqrt(sample_mean*(1-sample_mean)/n)
high = sample_mean + z*sqrt(sample_mean*(1-sample_mean)/n)
print('[{}; {}]'.format(low, high))
# print(_tconfint_generic(sample_mean, sample_std, n, 0.05, 'two-sided'))

print('2.3. Вильсоновский доверительный интервал для доли носителей варианта 13910T в популяции майя:')
wilson_interval = proportion_confint(count=1, nobs=50, method='wilson')
print('[{}; {}]'.format(wilson_interval[0], wilson_interval[1]))

print('2.5. Объём выборки для оценки нормального интервала с точностью до 0.01:')
n_samples = int(np.ceil(samplesize_confint_proportion(proportion=sample_mean,
                                                      half_length=0.01)))
print(n_samples)

array = np.arange(0, 1.01, 0.01)
print(array)
counts = list(map(lambda x: int(np.ceil(samplesize_confint_proportion(proportion=x,
                                                      half_length=0.01))), array))
print(counts)

print('2.5. Объём выборки необходимый в худшем случае для оценки нормального интервала:')
print('{} человек (при p = {})'.format(np.max(counts), array[np.argmax(counts)]))
plt.plot(array, counts)
plt.grid(True)
plt.show()

my_A = np.zeros(50)
my_A[0] = 1
print(my_A.std(ddof=1))
print(sqrt((49*(1/50)**2 + (49/50)**2)/(n-1)))
my_interval(my_A)