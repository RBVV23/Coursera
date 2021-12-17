import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

old_data = pd.read_csv('WAG_C_M.csv', sep=';', index_col=['month'], parse_dates=['month'],
                   dayfirst=True)
new_data = pd.read_csv('new_WAG_C_M.csv', sep=';', index_col=['month'], parse_dates=['month'],
                   dayfirst=True)

data = pd.concat([old_data, new_data])

print(data)

# print(data.columns)

