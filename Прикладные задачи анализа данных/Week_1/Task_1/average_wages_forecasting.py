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

data = pd.read_csv('WAG_C_M.csv', sep=';', index_col=['month'], parse_dates=['month'],
                   dayfirst=True)
# print(data.head())
# print(data.head(-5))
# print(data.columns)

data2 = pd.read_csv('Новый текстовый документ.csv', sep='\t')
print(data2.columns)

# data2 = data2.drop(['WAG_M', 'WAG_M_SA'])
data2.drop(['WAG_M', 'WAG_M_SA'], axis=1, inplace=True)

# print(data2.head())
print(data2.head(-5))
first_date = data2.month[0]
print('first_date = ', first_date)
data2.drop(['month'], axis=1, inplace=True)

date_list = [datetime.datetime.strptime("2016.01.01","%Y.%m.%d") + relativedelta(year=2016+(x)//12, month=1+(x)%12) for x in range(data2.shape[0])]
print(date_list[:5])
data2.index.name = 'month'
data2.index = list(map(lambda x: x.strftime("%Y.%m.%d"), date_list))
print(data2)
# first_date = datetime.datetime.strptime("2016-01-01","%Y-%m-%d")

data2.to_csv('new_WAG_C_M.csv', sep=';')

