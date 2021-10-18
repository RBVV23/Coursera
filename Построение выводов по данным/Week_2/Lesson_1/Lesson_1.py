import numpy as np
import pandas as pd

import scipy
from statsmodels.stats.weightstats import *
import matplotlib.pyplot as plt

data = pd.read_csv('ADHD.txt', sep=' ', header=0)
data.columns = ['Placebo', 'Methylphenidate']
data.plot.scatter('Placebo', 'Methylphenidate', c='r', s=30)
plt.grid()
plt.xlim((20,80))
plt.ylim((20,80))
plt.plot(range(100), c='black')
plt.show()