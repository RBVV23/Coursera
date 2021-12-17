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

# data = pd.read_csv('WAG_C_M.csv', sep=';', index_col=['month'], parse_dates=['month'],
#                    dayfirst=True)

old_data = pd.read_csv('WAG_C_M.csv', sep=';', index_col=['month'], parse_dates=['month'],
                   dayfirst=True)
new_data = pd.read_csv('new_WAG_C_M.csv', sep=';', index_col=['month'], parse_dates=['month'],
                   dayfirst=True)
data = pd.concat([old_data, new_data])

# data = data[0:297]

print(data)
print()

plt.figure(figsize=(12,7))
data['WAG_C_M'].plot()
plt.title('Исходные данные')
plt.ylabel('Средняя номинальная заработная плата')
# plt.show()

# print(data.columns)

data['WAG_C_M_boxcox'], lmbda = stats.boxcox(data['WAG_C_M'])
plt.figure(figsize=(12,7))
data.WAG_C_M_boxcox.plot()
plt.title('После преобразования Бокса-Кокса')
plt.ylabel('Средняя номинальная заработная плата')
# plt.show()

print('Оптимальный параметр преобразования Бокса-Кокса:')
print('\t \u03bb = ', round(lmbda, 6), '\n')

S = 12
data['WAG_C_M_boxcox_diff_S'] = data.WAG_C_M_boxcox - data.WAG_C_M_boxcox.shift(S)
sm.tsa.seasonal_decompose(data.WAG_C_M_boxcox_diff_S[S:]).plot()
# plt.show()

print('Критерий Дики-Фуллера (после преобразования Бокса-Кокса и сезонного дифференцирования):')
print('\tp-value = ', round(sm.tsa.stattools.adfuller(data.WAG_C_M_boxcox_diff_S[S:])[1], 6))
print('\t - согласно критерию ряд уже станционарный, но хотелось бы ещё сильнее избавиться от тренда', '\n')

data['WAG_C_M_boxcox_diff'] = data.WAG_C_M_boxcox_diff_S - data.WAG_C_M_boxcox_diff_S.shift(1)
sm.tsa.seasonal_decompose(data.WAG_C_M_boxcox_diff[(S+1):]).plot()
# plt.show()

print('Критерий Дики-Фуллера (после преобразования Бокса-Кокса, сезонного и обычного дифференцирований):')
print('\tp-value = ', round(sm.tsa.stattools.adfuller(data.WAG_C_M_boxcox_diff[(S+1):])[1], 6))
print('\t - гипотеза о нестационарности уверенно отвергается, тренд и остатки выглядят шумоподобно', '\n')


D, d = 1, 1
print('Количество сезонных и обычных дифференцирований:')
print('\t D = ', D)
print('\t d = ', d)

plt.figure(figsize=(12,7))
ax = plt.subplot(2,1,1)
sm.graphics.tsa.plot_acf(data.WAG_C_M_boxcox_diff[(S+1):].values.squeeze(), lags=4*S, ax=ax)
q, q_mark = 1, -0.1366
plt.plot(q, q_mark, marker='o', markerfacecolor=(1,1,0,0), markeredgecolor='r', markeredgewidth=3, markersize=15)
plt.text(q+0.1, 0.25, 'q = {}'.format(q))
Q, Q_mark = 0, 1
plt.plot(Q, Q_mark, marker='o', markerfacecolor=(1,1,0,0), markeredgecolor='r', markeredgewidth=3, markersize=15)
plt.text(Q+0.1, 0.85, 'Q\u00d7S = {}'.format(Q))

ax = plt.subplot(2,1,2)
sm.graphics.tsa.plot_pacf(data.WAG_C_M_boxcox_diff[(S+1):].values.squeeze(), lags=4*S, ax=ax)
p, p_mark = 10, -0.132213
plt.plot(p, p_mark, marker='o', markerfacecolor=(1,1,0,0), markeredgecolor='r', markeredgewidth=3, markersize=15)
plt.text(p+0.1, 0.15, 'p = {}'.format(p))
P, P_mark = 48, 0.1458
plt.plot(P, P_mark, marker='o', markerfacecolor=(1,1,0,0), markeredgecolor='r', markeredgewidth=3, markersize=15)
plt.text(P-1.2, 0.25, 'P\u00d7S = {}'.format(P))
plt.show()

Q = int(Q/S)
print('Начальные приближения из графика автокорреляционной функции:')
print('Q\u00d7S - номер последнего сезонного лага со значимой автокорреляцией')
print('q - номер последнего несезонного лага со значимой автокорреляцией')
print('\t Q = ', Q)
print('\t q = ', q)
P = int(P/S)
print('Начальные приближения из графика частичной автокорреляционной функции:')
print('P\u00d7S - номер последнего сезонного лага со значимой частичной автокорреляцией')
print('p - номер последнего несезонного лага со значимой автокорреляцией')
print('\t P = ', P)
print('\t p = ', p, '\n')


ps = range(p+1)
Ps = range(P+1)
qs = range(q+1)
Qs = range(Q+1)

parametrs = product(ps, qs, Ps, Qs)
parametrs_list = list(parametrs)
print('Количество моделей для перебора: ', len(parametrs_list), '\n')

# results = []
# best_aic = float('inf')
# warnings.filterwarnings('ignore')
#
# for i, param in enumerate(parametrs_list):
#     try:
#         model = sm.tsa.statespace.SARIMAX(data.WAG_C_M_boxcox, order=(param[0], d, param[1]),
#                                           seasonal_order=(param[2], D, param[3], S)).fit(disp=-1)
#     except ValueError:
#         print('Параметры невозможные для обучения: ', param)
#         continue
#     aic = model.aic
#     print('#',i)
#     if aic < best_aic:
#         best_model = model
#         best_aic = aic
#         best_param = param
#         print('\taic = {}'.format(aic))
#     results.append([param, model.aic])
#
# warnings.filterwarnings('default')
#
# result_table = pd.DataFrame(results)
# result_table.columns = ['parameters', 'aic']
# print(result_table.sort_values(by = 'aic', ascending=True).head())

# print()
# print(best_model.summary())
# print()