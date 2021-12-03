import pandas as pd
import numpy as np
from scipy.stats import probplot
import matplotlib.pyplot as plt

def my_get_boostraps_samples(data, n_samples):
    L = len(data)
    indices = np.random.randint(0, L, (n_samples, L))
    samples = data[indices]
    return samples
def my_stat_intervals(stat, alpha=0.05):
    low, high = np.percentile(stat, [100*alpha/2., 100*(1 - alpha/2.)])
    return low, high

data = pd.read_csv('ab_browser_test.csv', sep=',', header=0)

print(data.head())
print(data.info())

control = data[data.slot == 'control']
exp = data[data.slot == 'exp']

exp_clicks = exp.n_clicks
print('Количество кликов в группе "exp": ', np.sum(exp_clicks.values))

control_clicks = control.n_clicks
print('Количество кликов в группе "control": ', np.sum(control_clicks.values))

answer1 = round((np.sum(exp_clicks.values) - np.sum(control_clicks.values))/np.sum(control_clicks.values)*100, 3)
print('answer1 = ', answer1)

np.random.seed(0)

# btstrap_c = my_get_boostraps_samples(control_clicks.values, 500)
# meds_c = list(map(np.median, btstrap_c))
# means_c = list(map(np.mean, btstrap_c))
# btstrap_e = my_get_boostraps_samples(exp_clicks.values, 500)
# meds_e = list(map(np.median, btstrap_e))
# means_e = list(map(np.mean, btstrap_e))

## interval = my_stat_intervals(meds_c, 0.05)
## print('Доверительный интервал для медианы количества кликов в подгруппе "control":')
## print('[{}; {}]'.format(interval[0], interval[1]))

## interval = my_stat_intervals(meds_e, 0.05)
## print('Доверительный интервал для медианы количества кликов в подгруппе "exp":')
## print('[{}; {}]'.format(interval[0], interval[1]))

# meds_deltas = list(map(lambda x: x[0] - x[1], zip(meds_e, meds_c)))
# interval = my_stat_intervals(meds_deltas, 0.05)
# print('Доверительный интервал для разности медиан количества кликов в подгруппах "exp" и "control":')
# print('[{}; {}]'.format(interval[0], interval[1]))

## interval = my_stat_intervals(means_c, 0.05)
## print('Доверительный интервал для среднего количества кликов в подгруппе "control":')
## print('[{}; {}]'.format(interval[0], interval[1]))

## interval = my_stat_intervals(means_e, 0.05)
## print('Доверительный интервал для среднего количества кликов в подгруппе "exp":')
## print('[{}; {}]'.format(interval[0], interval[1]))

# means_deltas = list(map(lambda x: x[0] - x[1], zip(means_e, means_c)))
# interval = my_stat_intervals(means_deltas, 0.05)
# print('Доверительный интервал для разности средних количеств кликов в подгруппах "exp" и "control":')
# print('[{}; {}]'.format(interval[0], interval[1]))

np.random.seed(0)
n_boot_samples = 500
control_btstrap = my_get_boostraps_samples(control_clicks.values, n_boot_samples)
control_mean = list(map(np.mean, control_btstrap))
control_boot_chi_squared = []
i = 0
# control_btstrap = np.array([[1,2,3,3,2,1], [1,1,1,1,1,7]])

control_boot_chi_squared = np.sum(list(map(lambda x: (x - np.mean(x))**2, control_btstrap)), axis=1)
# print(len(qq))

# for btstrap in control_btstrap:
#     my_mean = np.mean(btstrap)
#     print(my_mean)
#     my_STDs = list(map(lambda x: (x - my_mean)**2, btstrap))
#     print(np.sum(my_STDs))
#     control_boot_chi_squared.append(np.sum(my_STDs))
    # i += 1
    # print(i)

plt.subplot(1,2,1)
N = len(control_mean)
probplot(control_mean, dist="norm", sparams=(N-1), plot=plt)
plt.subplot(1,2,2)
N = len(control_boot_chi_squared)
probplot(control_boot_chi_squared, dist="chi2", sparams=(N-1), plot=plt)
plt.show()