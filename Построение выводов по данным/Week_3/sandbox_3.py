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
Xm = np.array([1]*239 + [0]*515)
Xw = np.array([1]*203 + [0]*718)

MCC, a,b,c,d = my_MCC(Xm, Xw)
answer24 = round(MCC,4)
print('answer 2.4. = ', answer24)

print('2.5. Значимость отличия коэффициента корреляции Мэтьюса от нуля:')
# answer25 = round(answer25, 4)
print('p_value = ', scipy.stats.chi2_contingency(np.array([[a,b],[c,d]]))[1])
answer25 = 131
print('answer 2.5. = ', answer25)
