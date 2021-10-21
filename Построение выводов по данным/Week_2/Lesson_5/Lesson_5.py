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
print(seattle_data.head())

price2001 = seattle_data[seattle_data.Year == 2001].Price
price2002 = seattle_data[seattle_data.Year == 2002].Price

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.grid()
plt.hist(price2001, color='r', label='2001')
plt.legend()
plt.xlabel('Стоимость недвижимости')
plt.subplot(1,2,2)
plt.grid()
plt.hist(price2002, color='b', label='2002')
plt.legend()
plt.xlabel('Стоимость недвижимости')
plt.show()