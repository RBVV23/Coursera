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


def my_stat_intervals(stat, alpha):
    low, high = np.percentile(stat, [100*alpha/2., 100*(1 - alpha/2.)])
    return low, high
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
def my_proportions_confint_diff_ind(sample1, sample2, alpha=0.05):
    n1 = len(sample1)
    n2 = len(sample2)
    p1 = float(sum(sample1)/n1)
    p2 = float(sum(sample2)/n2)
    z = stats.norm.ppf(1-alpha/2.)
    low = p1-p2 - z*sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    high = p1-p2 + z*sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    return low, high
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
def my_get_boostraps_samples(data, n_samples):
    L = len(data)
    indices = np.random.randint(0, L, (n_samples, L))
    samples = data[indices]
    return samples
def my_odds(sample1, sample2):
    p1 = np.sum(sample1) / len(sample1)
    p2 = np.sum(sample2) / len(sample2)
    odds1 = float(p1 / (1 - p1))
    odds2 = float(p2 / (1 - p2))
    return float(odds1 / odds2)


# A = np.array([3,2,3,4,3])
# my_interval(A)


water = pd.read_csv('water.txt', sep='\t', header=0)
print(water.head())

my_X = water['mortality']

print('1.2. Постройте 95% доверительный интервал для средней годовой смертности в больших городах:')
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
answer12 = my_interval(my_X, precision=4)[0]
print('answer 1.2. = ',answer12)
#
print('1.3. Постройте 95% доверительный интервал для средней годовой смертности по всем южным городам:')
df = water[water['location'] == 'South']
my_X = df['mortality']
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
my_interval(df['mortality'], precision=4)
answer13 = my_interval(df['mortality'], precision=4)
print('answer 1.3. = ',answer13)
#
print('1.4. Постройте 95% доверительный интервал для средней годовой смертности по всем северным городам')
df = water[water['location'] == 'North']
my_X = df['mortality']
my_mean = my_X.mean()
my_std = my_X.std(ddof=1)/sqrt(len(my_X))
print(_tconfint_generic(my_mean, my_std, len(my_X) - 1, 0.05, 'two-sided'))
my_interval(df['mortality'], precision=4)
answer14 = my_interval(df['mortality'], precision=4)
print('answer 1.4. = ', answer14)


print('1.5. 95% доверительные интервалы для средней жёсткости воды в северных и южных городах:')

print(_tconfint_generic(water[water.location == 'North'].hardness.mean(), water[water.location == 'North'].hardness.std(ddof=1) / np.sqrt(len(water[water.location == 'North'])), len(water[water.location == 'North']) - 1, 0.05, 'two-sided'))
df = water[water['location'] == 'North']
my_interval(df['hardness'], precision=4)

print(_tconfint_generic(water[water.location == 'South'].hardness.mean(), water[water.location == 'South'].hardness.std(ddof=1) / np.sqrt(len(water[water.location == 'South'])), len(water[water.location == 'South']) - 1, 0.05, 'two-sided'))
df = water[water['location'] == 'South']
my_interval(df['hardness'], precision=4)
#
answer15 = my_interval(df['hardness'], precision=4)
print('answer15 = ' , answer15)
print('2.2. Нормальный доверительный интервал для доли носителей варианта 13910T в популяции майя:')

sample_mean = 1/50
n = 50
sample_std = sqrt((49*(1/50)**2 + (49/50)**2)/(n-1))
z = stats.norm.ppf(0.975)
low = sample_mean - z*sqrt(sample_mean*(1-sample_mean)/n)
high = sample_mean + z*sqrt(sample_mean*(1-sample_mean)/n)
print('[{}; {}]'.format(low, high))
print(_tconfint_generic(sample_mean, sample_std, n, 0.05, 'two-sided'))
answer22 = _tconfint_generic(sample_mean, sample_std, n, 0.05, 'two-sided')
print('answer 2.2. = ', answer22)
print('2.3. Вильсоновский доверительный интервал для доли носителей варианта 13910T в популяции майя:')
wilson_interval = proportion_confint(count=1, nobs=50, method='wilson')
print('[{}; {}]'.format(wilson_interval[0], wilson_interval[1]))

print('2.5. Объём выборки для оценки нормального интервала с точностью до 0.01:')
n_samples = int(np.ceil(samplesize_confint_proportion(proportion=sample_mean,
                                                      half_length=0.01)))
print(n_samples)
#
array = np.arange(0, 1.01, 0.01)
print(array)
counts = list(map(lambda x: int(np.ceil(samplesize_confint_proportion(proportion=x,
                                                      half_length=0.01))), array))
print(counts)
print('2.5. Объём выборки необходимый в худшем случае для оценки нормального интервала:')
print('{} человек (при p = {})'.format(np.max(counts), array[np.argmax(counts)]))
answer25 = np.max(counts)
print('answer 2.5. = ',answer25)
plt.plot(array, counts)
plt.grid(True)
# plt.show()

my_A = np.zeros(50)
my_A[0] = 1
print(my_A.std(ddof=1))
print(sqrt((49*(1/50)**2 + (49/50)**2)/(n-1)))
my_interval(my_A)

print('3.1. Уточненное правило 3-х ({}) сигм'.format(stats.norm.ppf((1 + 0.997)/2.)))
answer31 = round(stats.norm.ppf((1 + 0.997)/2.),4)
print('answer 3.1. = ',answer31)

n_asp = 11037
n_asp_inf = 104
n_plac = 11034
n_plac_inf = 189
group_asp = np.array([1]*n_asp_inf + [0]*(n_asp-n_asp_inf))
group_plac = np.array([1]*n_plac_inf + [0]*(n_plac-n_plac_inf))
print('3.5. Вероятность инфаркта снижается при приёме аспирина на величину:')
print(n_plac_inf/n_plac - n_asp_inf/n_asp)
answer35 = round(n_plac_inf/n_plac - n_asp_inf/n_asp,4)
print('answer 3.5. = ',answer35)

print('3.6. Доверительный интервал для снижения вероятности инфаркта при приёме аспирина:')
print(my_proportions_confint_diff_ind(group_plac, group_asp, alpha = 0.05))
answer36 = round(my_proportions_confint_diff_ind(group_plac, group_asp, alpha = 0.05)[1], 4)
print('answer 3.6. = ', answer36)

print('3.7. Шансы инфаркта при регулярном приёме аспирина понижаются в (раз):')
p_asp = n_asp_inf/n_asp
p_plac = n_plac_inf/n_plac
print('p_plac', p_plac)
odds_plac = p_plac/(1-p_plac)
print('odds_plac = ', odds_plac)
odds_asp = p_asp/(1-p_asp)
print('odds_asp = ', odds_asp)

answer37=odds_plac/odds_asp
print('answer 3.7. = ', answer37)

print(my_odds(group_plac, group_asp))

np.random.seed(0)

my_list = list(map(lambda x: my_odds(x[0],x[1]), zip(my_get_boostraps_samples(group_plac,1000),my_get_boostraps_samples(group_asp,1000))))
print('3.8. Доверительный интервал для шансов, построенный с помощью бутстрепа:')
print(my_stat_intervals(np.array(my_list), 0.05))
answer38 = round(my_stat_intervals(np.array(my_list), 0.05)[0],4)
print('answer 3.8. = ', answer38)

my_rest = np.array([1]*75 + [0]*(100-75))
all_rest = np.array([1]*67 + [0]*(100-67))

print('4.1. Достигаемый уровень значимости против двусторонней альтернативы:')
print(stats.binom_test(67, 100, 0.75, alternative = 'two-sided'))
answer41 = round(stats.binom_test(67, 100, 0.75, alternative = 'two-sided'),4)
print('answer 4.1. = ', answer41)

print(stats.binom_test(22, 50, 0.75, alternative = 'two-sided'))
# answer41 = round(stats.binom_test(67, 100, 0.75, alternative = 'two-sided'),4)

data = pd.read_csv('pines.txt', sep='\t', header=0)
print(data.head())

print('4.3. Среднее ожидаемое количество сосен в каждом квадрате')
answer43 = data.shape[0]/25
print('answer43 = ', answer43)

sn = data.sn
we = data.we


plt.scatter(sn, we, alpha=0.5, s=10)
plt.grid()
# plt.show()

binx = [0.0, 40, 80, 120, 160, 200]
biny = [0.0, 40, 80, 120, 160, 200]
ret = stats.binned_statistic_2d(sn, we, None, 'count', bins=[binx, biny])
print(ret[0])
expected = np.array([[answer43]*5]*5)
print(expected)

print('4.4. Значение статистики хи-квадрат:')
ret = stats.chisquare(ret[0].reshape(25), expected.reshape(25), ddof = 0)
print(ret)
answer44 = round(ret[0],2)
print('answer44 = ', answer44)






