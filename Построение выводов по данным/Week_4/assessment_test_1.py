import pandas as pd
import numpy as np


data = pd.read_csv('ab_browser_test.csv', sep=',', header=0)

print(data.head())
print(data.info())

exp_clicks = data[data.slot == 'exp'].n_clicks
print('Количество кликов в группе "exp": ', np.sum(exp_clicks.values))

control_clicks = data[data.slot == 'control'].n_clicks
print('Количество кликов в группе "control": ', np.sum(control_clicks.values))

# answer1 = exp_clicks - control_clicks