import pandas as pd
import numpy as np


data = pd.read_csv('ab_browser_test.csv', sep=',', header=0)

print(data.head())
print(data.info())

# exp_clicks = data[data.slot == 'exp'].shape[0]
# print('Количество кликов в группе "exp": ', exp_clicks)
exp_clicks = data[data.slot == 'exp'].n_clicks.value()
# print('Количество кликов в группе "control": ', control_clicks)
print(exp_clicks)

# answer1 = exp_clicks - control_clicks