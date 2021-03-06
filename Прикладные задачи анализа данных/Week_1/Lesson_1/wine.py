import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

def my_inv_boxcox(y, lmbda):
    if lmbda == 0:
        result = (np.exp(y))
    else:
        result = (np.exp(np.log(lmbda*y+1)/lmbda))
    return result

wine = pd.read_csv('monthly-australian-wine-sales.csv', sep=',', index_col=['month'], parse_dates=['month'],
                   dayfirst=True)
wine.sales = wine.sales * 1000

wine.sales.plot()
plt.ylabel('Продажи вина (л)')
plt.xlabel('Год')
# plt.show()

sm.tsa.seasonal_decompose(wine.sales).plot()
# plt.show()
print('Критерий Дики-Фуллера:')
print('\tp-value = ', round(sm.tsa.stattools.adfuller(wine.sales)[1], 6))
print('\t - ряд не стационарный, хоть мы и очень близки к порогу')


wine['sales_boxcox'], lmbda = stats.boxcox(wine.sales)
plt.figure(figsize=(15,8))
wine.sales_boxcox.plot()
plt.title('После преобразования Бокса-Кокса')
plt.ylabel('Продажи вина (л)')
# plt.show()
print('Оптимальный параметр преобразования Бокса-Кокса:')
print('\t \u03bb = ', round(lmbda, 6), '\n')
print('Критерий Дики-Фуллера (после преобразования Бокса-Кокса):')
print('\tp-value = ', round(sm.tsa.stattools.adfuller(wine.sales_boxcox)[1], 6))
print('\t - критерий отвергает нулевую гипотезу о нестационарности ряда, но мы отчётливо видим тренд', '\n')

S = 12
wine['sales_boxcox_diff_S'] = wine.sales_boxcox - wine.sales_boxcox.shift(S)
sm.tsa.seasonal_decompose(wine.sales_boxcox_diff_S[S:]).plot()
# plt.show()
print('Критерий Дики-Фуллера (после преобразования Бокса-Кокса и сезонного дифференцирования):')
print('\tp-value = ', round(sm.tsa.stattools.adfuller(wine.sales_boxcox_diff_S[S:])[1], 6))
print('\t - ряд всё ещё не стационарный', '\n')

wine['sales_boxcox_diff'] = wine.sales_boxcox_diff_S - wine.sales_boxcox_diff_S.shift(1)
sm.tsa.seasonal_decompose(wine.sales_boxcox_diff[(S+1):]).plot()
# plt.show()
print('Критерий Дики-Фуллера (после преобразования Бокса-Кокса, сезонного и обычного дифференцирований):')
print('\tp-value = ', round(sm.tsa.stattools.adfuller(wine.sales_boxcox_diff[(S+1):])[1], 6))
print('\t - гипотеза о нестационарности уверенно отвергается', '\n')


plt.figure(figsize=(15,8))
ax = plt.subplot(2,1,1)
sm.graphics.tsa.plot_acf(wine.sales_boxcox_diff[(S+1):].values.squeeze(), lags=4*S, ax=ax)
ax = plt.subplot(2,1,2)
sm.graphics.tsa.plot_pacf(wine.sales_boxcox_diff[(S+1):].values.squeeze(), lags=4*S, ax=ax)
# plt.show()

D, d = 1, 1
print('Количество сезонных и обычных дифференцирований:')
print('\t D = ', D)
print('\t d = ', d)

Q, q = int(12/S), 2
print('Начальные приближения из графика автокорреляционной функции:')
print('\t Q = ', Q)
print('\t q = ', q)
P, p = int(12/S), 4
print('Начальные приближения из графика частичной автокорреляционной функции:')
print('\t P = ', P)
print('\t p = ', p, '\n')

ps = range(p+1)
Ps = range(P+1)
qs = range(q+1)
Qs = range(Q+1)

parametrs = product(ps, qs, Ps, Qs)
parametrs_list = list(parametrs)
print('Количество моделей для перебора: ', len(parametrs_list), '\n')

results = []
best_aic = float('inf')
warnings.filterwarnings('ignore')

for param in parametrs_list:
    try:
        model = sm.tsa.statespace.SARIMAX(wine.sales_boxcox, order=(param[0], d, param[1]),
                                          seasonal_order=(param[2], D, param[3], S)).fit(disp=-1)
    except ValueError:
        print('Параметры невозможные для обучения: ', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

warnings.filterwarnings('default')

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by='aic', ascending=True).head())

print()
print(best_model.summary())
print()

plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
best_model.resid[(S+1):].plot()
plt.ylabel('Остатки')

ax = plt.subplot(2,1,2)
sm.graphics.tsa.plot_acf(best_model.resid[(S+1):].values.squeeze(), lags=4*S, ax=ax)
# plt.show()

print('Критерий Стьюдента для остатков модели:')
p_value = stats.ttest_1samp(best_model.resid[(S+1):], 0)[1]
print('\tp-value = ', round(p_value, 6))
print('\t - нулевая гипотеза о несмещенности остатков не отвергается')
print('Критерий Дики-Фуллера для остатков модели:')
p_value = sm.tsa.stattools.adfuller(best_model.resid[(S+1):])[1]
print('\tp-value = ', round(p_value, 6))
print('\t - нулевая гипотеза о нестационарности остатков отвергается')
print()

wine['model'] = my_inv_boxcox(best_model.fittedvalues, lmbda)
plt.figure(figsize=(15,8))
wine.sales.plot(label='исторические данные')
wine.model[(S+1):].plot(color='r', label='результаты моделирования')
plt.ylabel('Продажи вина (л)')
plt.legend()
# plt.show()

wine2 = wine[['sales']]
date_list = [datetime.datetime.strptime("1994-01-01","%Y-%m-%d") + relativedelta(year=1994+(8+x)//12, month=1+(8+x)%12)
             for x in range(36)]
print(date_list)
future = pd.DataFrame(index=date_list, columns=wine2.columns)
wine2 = pd.concat([wine2, future])
wine2['forecast'] = my_inv_boxcox(best_model.predict(start=wine.shape[0], end=wine.shape[0] + 35), lmbda)

plt.figure(figsize=(15,8))
wine2.sales.plot(label='моделирование')
wine2.forecast.plot(color='r', label='прогнозирование')
plt.ylabel('Продажи вина (л)')
plt.legend()
plt.show()







