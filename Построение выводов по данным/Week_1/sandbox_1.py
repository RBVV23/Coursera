import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
from math import sqrt
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic


def my_interval(X, alpha=0.95, precision=4, norm=True, flag=True):
    sample_mean=X.mean()
    sample_std = X.std(ddof=1)
    n = len(X)

    if flag:
        print('sample_mean = ', sample_mean)
        print('sample_std = ', sample_std)

    low = sample_mean - 2*sample_std/sqrt(n)
    low = round(low, precision)
    high = sample_mean + 2 * sample_std/sqrt(n)
    high = round(high, precision)
    print('Доверительный интервал {}%: ({} - {})'.format(100*alpha, low, high))
    print()
    return low, high

A = np.array([3,2,3,4,3])
my_interval(A)

water = pd.read_csv('water.txt', sep='\t', header=0)

print(water.head())

my_X = water['mortality']

print('2. Постройте 95% доверительный интервал для средней годовой смертности в больших городах:')
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
# my_interval(my_X, precision=4)

print('3. Постройте 95% доверительный интервал для средней годовой смертности по всем южным городам:')
df = water[water['location'] == 'South']
my_X = df['mortality']
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
# my_interval(df['mortality'], precision=4)

print('4. Постройте 95% доверительный интервал для средней годовой смертности по всем северным городам')
df = water[water['location'] == 'North']
my_X = df['mortality']
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
# my_interval(df['mortality'], precision=4)

print('5. 95% доверительные интервалы для средней жёсткости воды в северных и южных городах:')

print(_tconfint_generic(water[water.location == 'North'].hardness.mean(), water[water.location == 'North'].hardness.std(ddof=1) / np.sqrt(len(water[water.location == 'North'])), len(water[water.location == 'North']) - 1, 0.05, 'two-sided'))
df = water[water['location'] == 'North']
# my_interval(df['hardness'], precision=4)

print(_tconfint_generic(water[water.location == 'South'].hardness.mean(), water[water.location == 'South'].hardness.std(ddof=1) / np.sqrt(len(water[water.location == 'South'])), len(water[water.location == 'South']) - 1, 0.05, 'two-sided'))
df = water[water['location'] == 'South']
# my_interval(df['hardness'], precision=4)
