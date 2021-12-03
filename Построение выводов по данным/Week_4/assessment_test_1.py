import pandas as pd
import numpy as np


def my_get_boostraps_samples(data, n_samples):
    L = len(data)
    indices = np.random.randint(0, L, (n_samples, L))
    samples = data[indices]
    return samples


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

bstrap_0 = my_get_boostraps_samples(control.values, 100)
meds_0 = list(map(np.median, bstrap_0))
bstrap_1 = my_get_boostraps_samples(test.values, 100)
meds_1 = list(map(np.median, bstrap_1))