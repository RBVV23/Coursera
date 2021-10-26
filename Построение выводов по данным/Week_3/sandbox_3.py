import pandas as pd
from math import sqrt
from scipy import stats
from sklearn import model_selection, metrics, linear_model, ensemble
import itertools
import scipy


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