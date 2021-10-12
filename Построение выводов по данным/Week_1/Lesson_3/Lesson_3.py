import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint

def my_proportions_confint_diff_ind

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
#
# confidence interval: [-0.034157, 0.002157]
# def proportions_confint_diff_ind(sample1, sample2, alpha=0.05):
#     z = scipy.stats.norm.ppf(1 - alpha / 2.)
#     p1 = float(sum(sample1)) / len(sample1)
#     p2 = float(sum(sample2)) / len(sample2)
#
#     left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))
#     right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))
#
#     return (left_boundary, right_boundary)
# print("confidence interval: [%f, %f]" % proportions_confint_diff_ind(data.banner_a, data.banner_b))