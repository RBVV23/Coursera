from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.proportion import proportion_confint


def my_binom_zconfint(sample, p, alpha=0.05):
    q = 1 - p
    n = len(sample)
    mean = n*p
    variance = n*p*q
    z = z = stats.norm.ppf(1 - alpha/2.)
    low_number = int(round(mean - z*np.sqrt(variance)))
    low = np.sort(sample)[low_number]
    high_number = int(round(mean + z*np.sqrt(variance)))
    high =  np.sort(sample)[high_number]
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
def my_proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    p1 = sum(sample1) / n1
    n2 = len(sample2)
    p2 = sum(sample2) / n2
    P = (p1 * n1 + p2 * n2) / (n1 + n2)
    z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1 / n1 + 1 / n2))
    return z_stat

data = pd.read_csv('credit_card_default_analysis.csv')

pd.set_option('display.max_columns', 25)

print(data.head())

plt.figure(figsize=(16,9))
plt.subplot(121)
plt.title('Гистограмма размера кредитного лимита')
plt.hist(data['LIMIT_BAL'], edgecolor='k')
plt.grid()


control = data[data.default == 0]['LIMIT_BAL']
control_median = np.median(control['LIMIT_BAL'])
print('control_median = ', control_median)

test = data[data.default == 1]['LIMIT_BAL']
test_median = np.median(test['LIMIT_BAL'])
print('test_median = ', test_median)


plt.subplot(122)
plt.title('Гистограмма размера кредитного лимита (вернули/не вернули)')
plt.hist(control, label='Вернули кредит', edgecolor='k')
plt.hist(test, label='Не вернули кредит', edgecolor='k')
plt.grid()
plt.legend()
plt.show()


interval = my_binom_zconfint(control, 0.5)
print('Доверительный интервал для медианы среди вернувших кредит:')
print('[{}; {}]'.format(interval[0], interval[1]))

interval = my_binom_zconfint(test, 0.5)
print('Доверительный интервал для медианы среди не вернувших кредит:')
print('[{}; {}]'.format(interval[0], interval[1]))

res = stats.mannwhitneyu(control, test)
print(res)
print('p-value = ', res[1])


plt.figure(figsize=(16,9))
plt.subplot(121)
plt.title('Половая гистограмма заёмщиков')
plt.hist(data.SEX, edgecolor='k')
plt.grid(axis='y')

men = data[data.SEX == 1]['default']
women = data[data.SEX == 2]['default']

plt.title('Гистограмма невозвратов кредитов (по половому признаку)')
plt.subplot(122)
plt.hist(data.SEX[data.default == 0], label='Вернули', edgecolor='k')
plt.hist(data.SEX[data.default == 1], label='Не вернули', edgecolor='k')
plt.grid(axis='y')
plt.legend()
plt.show()



sample = np.abs(data.SEX[data.default == 0] - 2)
interval = np.round(proportion_confint(count=sum(sample), nobs=len(sample),
                                     method='wilson'),4)

print('Доверительный интервал для доли мужчин среди вернувших:')
print('[{}; {}]'.format(interval[0], interval[1]))

sample = np.abs(data.SEX[data.default == 1] - 2)
interval = np.round(proportion_confint(count=sum(sample), nobs=len(sample),
                                     method='wilson'),4)
print('Доверительный интервал для доли мужчин среди не вернувших:')
print('[{}; {}]'.format(interval[0], interval[1]))


sample1 = np.abs(data.SEX[data.default == 0] - 2)
sample2 = np.abs(data.SEX[data.default == 1] - 2)

interval = np.round(my_proportions_confint_diff_ind(sample1, sample2, alpha=0.05),4)

print('Доверительный интервал разности долей мужчин среди вернувших и не вернувших:')
print('[{}; {}]'.format(interval[0], interval[1]))

sample1 = np.abs(data.SEX[data.default == 0] - 2)
sample2 = np.abs(data.SEX[data.default == 1] - 2)

z_stat = my_proportions_diff_z_stat_ind(sample1, sample2)
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print('p-value = ', p_value)