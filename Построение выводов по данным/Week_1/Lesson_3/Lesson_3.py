import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint
from math import sqrt

def my_proportions_confint_diff_ind(sample1, sample2, alpha=0.05):
    n1 = len(sample1)
    n2 = len(sample2)
    p1 = float(sum(sample1)/n1)
    p2 = float(sum(sample2)/n2)
    z = stats.norm.ppf(1-alpha/2.)
    low = p1-p2 - z*sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    high = p1-p2 + z*sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
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
    # print('f = ',f)
    # print('g = ',g)
    low = float(f - g)/n - z*sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    high = float(f - g)/n + z*sqrt(float((f + g)) / n**2 - float((f - g))**2 / n**3)
    return low, high

data = pd.read_csv('banner_click_stat.txt', header=None, sep='\t')
data.columns = ['banner_a', 'banner_b']

print(data.head())
print(data.describe())

conf_interval_banner_a = proportion_confint(count=sum(data.banner_a), nobs=data.shape[0], method='wilson')
conf_interval_banner_b = proportion_confint(count=sum(data.banner_b), nobs=data.shape[0], method='wilson')

print('Вильсоновский интервал [{}; {}] для баннера "A"'.format(conf_interval_banner_a[0],
                                                            conf_interval_banner_a[1]))
print('Вильсоновский интервал [{}; {}] для баннера "B"'.format(conf_interval_banner_b[0],
                                                            conf_interval_banner_b[1]))

bound = my_proportions_confint_diff_ind(data.banner_a, data.banner_b)
print('Доверительный интервал для разности долей (несвязанные выборки): [{}; {}]'.format(bound[0], bound[1]))

new_bound = my_proportions_confint_diff_rel(data.banner_a, data.banner_b)
# new_bound = my_proportions_confint_diff_rel(np.array([1,2,3]), np.array([10,20,30]))
print('Доверительный интервал для разности долей (связанные выборки): [{}; {}]'.format(new_bound[0], new_bound[1]))
