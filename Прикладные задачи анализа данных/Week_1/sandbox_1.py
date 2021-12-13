from scipy import stats
import pandas as pd
# print(pd.__version__)
import numpy as np
# print(np.__version__)
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv('monthly-milk-production.csv', sep=';', header=0,
                   parse_dates=['month'], dayfirst=True)
print(data)

plt.figure()
plt.title('milk')
plt.plot(data.milk.values)
# plt.show()

res = sm.tsa.stattools.adfuller(data.milk.values)
print(res)
answer14 = round(res[1],2)
print('answer 1.4 = ', answer14)

data['n_days'] = list(map(lambda x: x.days_in_month, data.month))
data['avg_per_day'] = data.milk / data.n_days
# print(data.head())

plt.figure()
plt.title('average milk per day')
plt.plot(data.avg_per_day.values)
plt.show()

answer15 = round(np.sum(data.avg_per_day.values), 2)
print('answer 1.5 = ', answer15)