import pandas as pd

from math import sqrt
from scipy import stats

from sklearn import model_selection, metrics, linear_model, ensemble
import itertools
import scipy
from statsmodels.stats.weightstats import *
from math import factorial

def my_proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    p1 = sum(sample1) / n1
    n2 = len(sample2)
    p2 = sum(sample2) / n2
    P = (p1 * n1 + p2 * n2) / (n1 + n2)
    Z = (p1 - p2) / np.sqrt(P * (1 - P) * (1 / n1 + 1 / n2))
    return Z
def my_proportions_diff_z_test(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Недопустимое значения параметра "alternative"\n'
                         'допустимо: "two-sided", "less" или "greater"')

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(abs(z_stat)))
    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)
    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)
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

def my_get_boostraps_samples(data, n_samples):
    L = len(data)
    indices = np.random.randint(0, L, (n_samples, L))
    samples = data[indices]
    return samples
def my_stat_intervals(stat, alpha=0.05):
    low, high = np.percentile(stat, [100*alpha/2., 100*(1 - alpha/2.)])
    return low, high

def my_permutation_test(sample1, sample2, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError('Недопустимое значения параметра "alternative"\n'
                         'допустимо: "two-sided", "less" или "greater"')
    t_stat = my_permutation_t_stat_ind(sample1, sample2)
    zero_distr = my_permutation_zero_dist_ind(sample1, sample2, max_permutations)

    if alternative == 'two-sided':
        res = sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr])/len(zero_distr)
    if alternative == 'less':
        res = sum([1. if x <= t_stat else 0. for x in zero_distr])/len(zero_distr)
    if alternative == 'greater':
        res = sum([1. if x >= t_stat else 0. for x in zero_distr])/len(zero_distr)
    return res
def my_get_random_combinations(n1, n2, max_combinations):
    index = list(range(n1 + n2))
    indices = set([tuple(index)])
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    result = [(index[:n1], index[n1:]) for index in indices]
    return result
def my_permutation_zero_dist_ind(sample1, sample2, max_combinations = None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n2 = len(sample2)
    n = len(joined_sample)

    if max_combinations:
        indices = my_get_random_combinations(n1, n2, max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) for index in itertools.combinations(range(n), n1)]
    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() for i in indices]
    return distr
def my_permutation_t_stat_ind(sample1, sample2):
    result = np.mean(sample1) - np.mean(sample2)
    return result
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
def my_p_value(expect_mean=9.5, std=0.4, n=160, sample_mean=9.57, alpha=0.95, alternative='two-sided'):
    z = (sample_mean - expect_mean)/(std/sqrt(n))
    Fz = stats.t.ppf(0.05/2,n-1)
    S = 0.5*(1 + scipy.special.erf((z - 0)/sqrt(2*1**2)))
    if alternative == 'two-sided':
        p = 2*(1 - scipy.stats.norm.cdf(abs(z)))
    if alternative == 'less':
        p = scipy.stats.norm.cdf(z)
    if alternative == 'greater':
        p = 1 - scipy.stats.norm.cdf(z)
    return p
def my_odds(sample1, sample2):
    p1 = np.sum(sample1) / len(sample1)
    p2 = np.sum(sample2) / len(sample2)
    odds1 = float(p1 / (1 - p1))
    odds2 = float(p2 / (1 - p2))
    return float(odds1 / odds2)


A = np.array([1,2,3,4,5,6,7,8,9,10])
print('np.std() = ', A.std())
print('my.std = ', np.sqrt(np.sum((A - A.mean())**2)/len(A)) )
print('np.std(ddof=1) = ', A.std(ddof=1))
print('my.std = ', np.sqrt(np.sum((A - A.mean())**2)/(len(A)-1)) )

prec=15

def my_hypergeometric_function(a,b,c,z, precision=15, max_iters=50000):
    eps = 0.1**(precision+1)
    sum = float(0)
    k = 1
    iter = 1
    delta = eps+1
    while delta > eps and iter < max_iters:
        mult = 1
        for l in range(k):
            mult *= (a+l)*(b+l) / ((1+l)*(c+l))
        delta = mult * z**k
        sum += delta
        k += 1
        iter += 1
        # print(iter)
    return round((sum + 1),precision)

n = 1000
t = 0.5
print(my_hypergeometric_function(0.5, 0.5*(n+1), 3/2, -(t**2)/n))

x = 1
print('arcsin(x)/x = ', round(np.arcsin(x)/x, prec))
print('F = ', my_hypergeometric_function(1/2, 1/2, 3/2, x**2))

def my_gamma_function(z, precision=15):
    eps = 0.1**(precision+1)
    mult = float(1)
    n = float(1)
    delta = eps+1
    while delta > eps:
        new_mult = mult * ((1 + 1/n)**z / (1 + z/n))
        delta = abs(new_mult - mult)
        # print(delta)
        mult = new_mult
        n += 1
    return round(float(mult/z),precision)

print('Г(5) = ', factorial(4))
print('Г(5) = ', my_gamma_function(5))
