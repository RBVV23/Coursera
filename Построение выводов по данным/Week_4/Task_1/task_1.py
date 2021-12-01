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
def my_v_cramer(table):
    K1 = table.shape[0]
    K2 = table.shape[1]
    N = np.sum(table)
    minK = min(K1, K2)
    chi2 = stats.chi2_contingency(table)[0]
    return np.sqrt(chi2 / (N*(minK-1)))
def my_get_boostraps_samples(data, n_samples):
    L = len(data)
    indices = np.random.randint(0, L, (n_samples, L))
    samples = data[indices]
    return samples
def my_stat_intervals(stat, alpha=0.05):
    low, high = np.percentile(stat, [100*alpha/2., 100*(1 - alpha/2.)])
    return low, high
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

plt.figure(figsize=(16,9))
plt.subplot(121)
plt.title('Гистограмма образования заёмщиков')
plt.hist(data.EDUCATION, bins=7, edgecolor='k')
plt.grid(axis='y')

plt.subplot(122)
plt.title('Гистограмма образования заёмщиков (вернули/не вернули)')
plt.hist(data.EDUCATION[data.default == 0], label='Вернули', bins=7, edgecolor='k')
plt.hist(data.EDUCATION[data.default == 1], label='Не вернули', bins=7,  edgecolor='k')
plt.legend()
plt.grid(axis='y')
plt.show()

proportions = []

for edu in range(data.EDUCATION.value_counts().shape[0]):
    def_0 = data[(data.EDUCATION == edu) & (data.default == 0)].shape[0]
    alls = data[data.EDUCATION == edu].shape[0]
    prop = def_0/alls
    print('Уровень образования "{}", доля возвратов: {}%'.format(edu, round(prop*100,1)))
    proportions.append(prop)

plt.figure(figsize=(16,9))
plt.title('Зависимость доли возвращённых кредитов от уровня образования')
plt.plot(proportions, linewidth=5, color='r')
plt.grid()
plt.axis([0, 6, 0.5, 1])
plt.show()

print('Средняя доля возвратов: {}% \n'.format(round(np.mean(proportions)*100),1))

for edu in range(data.EDUCATION.value_counts().shape[0]):
    def_0 = data[(data.EDUCATION == edu) & (data.default == 0)].shape[0]
    alls = data[data.EDUCATION == edu].shape[0]
    print('Уровень образования "{}":'.format(edu))
    print('\t', '{} - всего'.format(alls))
    print('\t', '{} - вернули'.format(def_0))
    print('\t', '{} - не вернули'.format(alls - def_0))

mean_prop = np.mean(proportions)
proportions_exp = []
proportions_obs = []

for edu in range(data.EDUCATION.value_counts().shape[0]):
    def_0 = data[(data.EDUCATION == edu) & (data.default == 0)].shape[0]
    alls = data[data.EDUCATION == edu].shape[0]
    expect = alls*mean_prop
    proportions_exp.append(expect)
    proportions_obs.append(def_0)
    print('Уровень образования "{}":'.format(edu))
    print('\t', '{} - вернули (налбюдение)'.format(def_0))
    print('\t', '{} - не вернули (ожидание)'.format(int(round(expect,0))))
    print('\t', '{} - разность наблюдения и ожидания'.format(int(def_0 - expect)))

res = stats.chisquare(f_obs=proportions_obs, f_exp=proportions_exp, ddof=1)
print('p-value = ', res[1])

plt.figure(figsize=(16,9))
subplot(121)
plt.title('Гистограмма заёмщиков по семейному положению')
plt.hist(data.MARRIAGE, bins=4, edgecolor='k')
plt.grid(axis='y')


plt.subplot(122)
plt.title('Гистограмма заёмщиков по семейному положению (вернули/не вернули)')
plt.hist(data.MARRIAGE[data.default == 0], bins=4, label='Вернули', edgecolor='k')
plt.hist(data.MARRIAGE[data.default == 1], bins=4, label='Не вернули', edgecolor='k')
plt.grid(axis='y')
plt.legend()
plt.show()

table = np.zeros((data.MARRIAGE.value_counts().shape[0], data.default.value_counts().shape[0]))

for x in range(data.default.value_counts().shape[0]):
    for y in range(data.MARRIAGE.value_counts().shape[0]):
        control = data[(data.default == x) & (data.MARRIAGE == y)]
        table[y, x] = control.shape[0]

print('Построим таблицу сопряженности:')
print(table)

print('Коэффициент V Крамера: ', round(my_v_cramer(table),4))

plt.figure(figsize=(16,9))
subplot(121)
plt.title('Гистограмма заёмщиков по возрасту')
plt.hist(data.AGE, edgecolor='k')
plt.grid(axis='y')


plt.subplot(122)
plt.title('Гистограмма заёмщиков по возрасту (вернули/не вернули)')
plt.hist(data.AGE[data.default == 0], label='Вернули', edgecolor='k')
plt.hist(data.AGE[data.default == 1], label='Не вернули', edgecolor='k')
plt.grid(axis='y')
plt.legend()
plt.show()


control = data.AGE[data.default == 0]
print('control_median = ', np.median(control))

test = data.AGE[data.default == 1]
print('test_median = ', np.median(test))

np.random.seed(0)

bstrap_0 = my_get_boostraps_samples(control.values, 100)
meds_0 = list(map(np.median, bstrap_0))
bstrap_1 = my_get_boostraps_samples(test.values, 100)
meds_1 = list(map(np.median, bstrap_1))

interval = my_stat_intervals(meds_0, 0.05)
print('Доверительный интервал для медианы возраста среди вернувших кредит:')
print('[{}; {}]'.format(interval[0], interval[1]))

interval = my_stat_intervals(meds_1, 0.05)
print('Доверительный интервал для медианы возраста среди не вернувших кредит:')
print('[{}; {}]'.format(interval[0], interval[1]))

delta_meds = list(map(lambda x: x[0] - x[1], zip(meds_0, meds_1)))

interval = np.round(my_stat_intervals(delta_meds, 0.05))
print('Доверительный интервал для разности медиан возрастов среди вернувших и не вернувших кредит:')
print('[{}; {}]'.format(interval[0], interval[1]))

p_value = my_permutation_test(control, test, 1000)
print('p_value = ', p_value)
