import pandas as pd
from math import sqrt
import numpy as np
from scipy import stats
from sklearn import model_selection, metrics, linear_model, ensemble
import itertools
import scipy

def my_MCC(X1, X2):
    a = 0
    b = 0
    c = 0
    d = 0
    for x1, x2 in zip(X1,X2):
        if x1 == x2:
            if x1 == 0:
                a += 1
            else:
                d += 1
        else:
            if x1 == 0:
                b += 1
            else:
                c += 1
    MCC = (a*d - b*c)/sqrt((a+b)*(a+c)*(b+d)*(c+d))
    return MCC, a,b,c,d

data = pd.read_csv('illiteracy.txt', sep='\t', header=0)
print(data.head())
print(data.describe())

print(data.corr())

print('1.4. Выборочный коэффициент корреляции Пирсона между долей неграмотных женщин и рождаемостью:')
answer14 = round(data.corr()['Births']['Illit'],4)
print('answer 1.4. = ', answer14)

print('1.5. Выборочный коэффициент корреляции Спирмена между долей неграмотных женщин и рождаемостью:')
answer15 = round(data.corr(method='spearman')['Births']['Illit'],4)
print('answer 1.5. = ', answer15)

data = pd.read_csv('water.txt', sep='\t', header=0)
print(data.head())
print(data.describe())

print(data.corr())

print('2.1. Значение коэффициента корреляции Пирсона между жесткостью воды и смертностью:')
answer21 = round(data.corr(method='pearson')['mortality']['hardness'],4)
print('answer 2.1. = ', answer21)

print('2.2. Значение коэффициента корреляции Спирмена между жесткостью воды и смертностью:')
answer22 = round(data.corr(method='spearman')['mortality']['hardness'],4)
print('answer 2.2. = ', answer22)

corr1 = data[data['location'] == 'South'].corr(method='pearson')['mortality']['hardness']
corr2 = data[data['location'] == 'North'].corr(method='pearson')['mortality']['hardness']
print('2.3. Значение коэффициента корреляции Пирсона между жесткостью воды и смертностью отдельно для южных и северных городов:')
print('South: ', corr1)
print('North: ', corr2)
if abs(corr1) < abs(corr2):
    answer23 = round(corr1,4)
else:
    answer23 = round(corr2,4)
print('answer 2.3. = ', answer23)

print('2.4. Значение коэффициента корреляции Мэтьюса между полом и частотой похода в бары:')
a = 239
b = 203
c = 515
d = 718

MCC = (a*d - b*c)/sqrt((a+b)*(a+c)*(b+d)*(c+d))
answer24 = round(MCC,4)
print('answer 2.4. = ', answer24)

print('2.5. Значимость отличия коэффициента корреляции Мэтьюса от нуля:')
print('p_value = ', scipy.stats.chi2_contingency(np.array([[a,b],[c,d]]))[1])
answer25 = 5
print('answer 2.5. = ', answer25)

print('2.6. Доверительный интервал для разности долей мужчин и женщин, относительно часто бывающих в барах:')
z = scipy.stats.norm.ppf(1 - 0.05 / 2.)
n1 = a + c
p1 = a/n1
n2 = b + d
p2 = b/n2
delta = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
low = p1 - p2 - z * delta
high = p1 - p2 + z * delta
answer26 = round(low, 4)
print('answer 2.6. = ', answer26)

print('2.7. Достигаемый уровень значимости для гипотезы о равенстве долей среди мужчин и женщин:')
P = float(p1 * n1 + p2 * n2) / (n1 + n2)
z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))
p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))
print('p_value = ', p_value)
answer27 = 6
print('answer 2.7. = ', answer27)

print('2.8. Значение статистики кси-квадрат:')
table = np.array([[197,111,33],[382, 685, 331], [110, 342, 333]])
answer28 = scipy.stats.chi2_contingency(table)[0]
answer28 = round(answer28, 4)
print('answer 2.8. = ', answer28)

print('2.9. Достигаемый уровень значимости:')
print('p_value = ', scipy.stats.chi2_contingency(table)[1])
answer29 = 62
print('answer 2.9. = ', answer29)

print('2.10. Значение коэффициента V Крамера для рассматриваемых признаков:')
n = np.sum(table)
K1 = 3
K2 = 3
answer210 = sqrt(scipy.stats.chi2_contingency(table)[0]/(n*(min(K1,K2) - 1)))
answer210 = round(answer210,4)
print('answer 2.10. = ', answer210)
