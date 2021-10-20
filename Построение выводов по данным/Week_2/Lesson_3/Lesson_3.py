import numpy as np
import pandas as pd
import itertools

from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint

import matplotlib.pyplot as plt

mouses_data = pd.read_csv('mirrow_mouses.txt', header=None)
mouses_data.columns = ['proportion_of_time']

print(mouses_data)