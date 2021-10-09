import pandas as pd
# print(pd.__version__)
import numpy as np
# print(np.__version__)
import matplotlib.pyplot as plt
from conda_verify.utilities import sel_pat


def my_interval(X, alpha=0.95, norm=True):
    sample_mean=X.mean()
    print('sample_mean = ', sample_mean)
    sample_std = X.std(ddof=1)
    print('sample_std = ', sample_std)
    low = sample_mean - 2*sample_std
    high = sample_mean + 2 * sample_std
    print('Доверительный интервал {}%: ({} - {})'.format(100*alpha, low, high))
    return low, high

A = np.array([3,2,3,4,3])
# my_interval(A)

water = pd.read_csv('water.txt', sep='\t', header=0)

print(water.head())

print(water.info())