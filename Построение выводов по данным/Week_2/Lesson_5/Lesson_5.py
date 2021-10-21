import numpy as np
import pandas as pd
import itertools

from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.weightstats import *
import matplotlib.pyplot as plt

seattle_data = pd.read_csv('seattle.txt', header=0, sep='\t')
print('seattle_data.shape = ', seattle_data.shape)