import scipy
import pandas as pd
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint


def my_proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    n1 = len(sample1)
    p1 = sum(sample1)/n1
    n2 = len(sample2)
    p2= sum(sample2)/n2
    delta = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    low = p1 - p2 - z*delta
    high = p1 - p2 + z*delta
    return low, high
def my_proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    p1 = sum(sample1) / n1
    n2 = len(sample2)
    p2 = sum(sample2) / n2
    P = (p1*n1 + p2*n2) / (n1 + n2)
    Z = (p1 - p2)/np.sqrt(P*(1-P)*(1/n1 + 1/n2))
    return Z
def my_proportions_diff_z_test(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Недопустимое значения параметра "alternative"\n'
                         'допустимо: "two-sided", "less" или "greater"')
    
    if alternative == 'two-sided':
        return 2*(1 - scipy.stats.norm.cdf(abs(z_stat)))
    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)
    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)
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
    low = float(f - g)/n - z*np.sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    high = float(f - g)/n + z*np.sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    return low, high
def my_proportions_diff_z_stat_rel(sample1, sample2):
    sample = list(zip(sample1, sample2))
    n = len(sample)
    f = 0
    g = 0
    for x in sample:
        if x[0] == 1 and x[1] == 0:
            f += 1
        if x[0] == 0 and x[1] == 1:
            g += 1
    result = (f-g)/np.sqrt(f+g - ((f-g)**2)/n)
    return result


data = pd.read_csv('banner_click_stat.txt', header=None, sep='\t')
data.columns = ['banner_a', 'banner_b']

print(data.head())
print(data.describe())

conf_interval_banner_a = proportion_confint(sum(data.banner_a), data.shape[0], method='wilson')
print('Доверительный интервал доли кликов (95%) для баннера "A": [{}; {}]'.format(conf_interval_banner_a[0],
                                                                                  conf_interval_banner_a[1]))
conf_interval_banner_b = proportion_confint(sum(data.banner_b), data.shape[0], method='wilson')
print('Доверительный интервал доли кликов (95%) для баннера "B": [{}; {}]'.format(conf_interval_banner_b[0],
                                                                                  conf_interval_banner_b[1]))

interval = my_proportions_diff_confint_ind(data.banner_a, data.banner_b)
print('Доверительный интервал для разности долей кликов (95%): [{}; {}]'.format(interval[0],
                                                                                  interval[1]))

print('Значение p-value (двустроронняя альтернатива): ', my_proportions_diff_z_test(my_proportions_diff_z_stat_ind(data.banner_a,
                                                                                      data.banner_b)))
print('Значение p-value (односторонняя альтернатива): ', my_proportions_diff_z_test(my_proportions_diff_z_stat_ind(data.banner_a,
                                                                                      data.banner_b), 'less'))

interval = my_proportions_confint_diff_rel(data.banner_a, data.banner_b)
print('Доверительный интервал для разности долей кликов с учётом связи выборок (95%): [{}; {}]'.format(interval[0],
                                                                                  interval[1]))

print('Значение p-value с учётом связи выборок (двустроронняя альтернатива): ', my_proportions_diff_z_test(my_proportions_diff_z_stat_rel(data.banner_a,
                                                                                      data.banner_b)))
print('Значение p-value с учётом связи выборок (односторонняя альтернатива): ', my_proportions_diff_z_test(my_proportions_diff_z_stat_rel(data.banner_a,
                                                                                      data.banner_b), 'less'))


