import pandas as pd
import numpy as np


data = pd.read_csv('ab_browser_test.csv')

print(data.head())
print(data.info())

exp_clicks = data[data.slot == 'exp'].shape[0]
print('Количество кликов в группе "exp": ', exp_clicks)
control_clicks = data[data.slot == 'control'].n_clicks
print('Количество кликов в группе "control": ', control_clicks)

# answer1 = exp_clicks - control_clicks