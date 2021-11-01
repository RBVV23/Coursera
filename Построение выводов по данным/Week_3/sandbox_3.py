import pandas as pd
from math import sqrt
import numpy as np
from scipy import stats
import itertools
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

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


# data = pd.read_csv('illiteracy.txt', sep='\t', header=0)
# print(data.head())
# print(data.describe())
#
# print(data.corr())
#
# print('1.4. Выборочный коэффициент корреляции Пирсона между долей неграмотных женщин и рождаемостью:')
# answer14 = round(data.corr()['Births']['Illit'],4)
# print('answer 1.4. = ', answer14)
#
# print('1.5. Выборочный коэффициент корреляции Спирмена между долей неграмотных женщин и рождаемостью:')
# answer15 = round(data.corr(method='spearman')['Births']['Illit'],4)
# print('answer 1.5. = ', answer15)
#
# data = pd.read_csv('water.txt', sep='\t', header=0)
# print(data.head())
# print(data.describe())
#
# print(data.corr())
#
# print('2.1. Значение коэффициента корреляции Пирсона между жесткостью воды и смертностью:')
# answer21 = round(data.corr(method='pearson')['mortality']['hardness'],4)
# print('answer 2.1. = ', answer21)
#
# print('2.2. Значение коэффициента корреляции Спирмена между жесткостью воды и смертностью:')
# answer22 = round(data.corr(method='spearman')['mortality']['hardness'],4)
# print('answer 2.2. = ', answer22)
#
# corr1 = data[data['location'] == 'South'].corr(method='pearson')['mortality']['hardness']
# corr2 = data[data['location'] == 'North'].corr(method='pearson')['mortality']['hardness']
# print('2.3. Значение коэффициента корреляции Пирсона между жесткостью воды и смертностью отдельно для южных и северных городов:')
# print('South: ', corr1)
# print('North: ', corr2)
# if abs(corr1) < abs(corr2):
#     answer23 = round(corr1,4)
# else:
#     answer23 = round(corr2,4)
# print('answer 2.3. = ', answer23)
#
# print('2.4. Значение коэффициента корреляции Мэтьюса между полом и частотой похода в бары:')
# a = 239
# b = 203
# c = 515
# d = 718
#
# MCC = (a*d - b*c)/sqrt((a+b)*(a+c)*(b+d)*(c+d))
# answer24 = round(MCC,4)
# print('answer 2.4. = ', answer24)
#
# print('2.5. Значимость отличия коэффициента корреляции Мэтьюса от нуля:')
# print('p_value = ', scipy.stats.chi2_contingency(np.array([[a,b],[c,d]]))[1])
# answer25 = 5
# print('answer 2.5. = ', answer25)
#
# print('2.6. Доверительный интервал для разности долей мужчин и женщин, относительно часто бывающих в барах:')
# z = scipy.stats.norm.ppf(1 - 0.05 / 2.)
# n1 = a + c
# p1 = a/n1
# n2 = b + d
# p2 = b / n2
# delta = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
# low = p1 - p2 - z * delta
# high = p1 - p2 + z * delta
# answer26 = round(low, 4)
# print('answer 2.6. = ', answer26)
#
# print('2.7. Достигаемый уровень значимости для гипотезы о равенстве долей среди мужчин и женщин:')
# P = float(p1 * n1 + p2 * n2) / (n1 + n2)
# z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))
# p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))
# print('p_value = ', p_value)
# answer27 = 6
# print('answer 2.7. = ', answer27)
#
# print('2.8. Значение статистики кси-квадрат:')
# table = np.array([[197,111,33],[382, 685, 331], [110, 342, 333]])
# answer28 = scipy.stats.chi2_contingency(table)[0]
# answer28 = round(answer28, 4)
# print('answer 2.8. = ', answer28)
#
# print('2.9. Достигаемый уровень значимости:')
# print('p_value = ', scipy.stats.chi2_contingency(table)[1])
# answer29 = 62
# print('answer 2.9. = ', answer29)
#
# print('2.10. Значение коэффициента V Крамера для рассматриваемых признаков:')
# n = np.sum(table)
# K1 = 3
# K2 = 3
# answer210 = sqrt(scipy.stats.chi2_contingency(table)[0]/(n*(min(K1,K2) - 1)))
# answer210 = round(answer210,4)
# print('answer 2.10. = ', answer210)
#
# print('4.3. Количество статистически значимых на уровне 0.05 различий:')
#
# data = pd.read_csv('AUCs.txt', sep='\t', header=0)
#
# print(data.head())
#
# my_columns = data.columns.values[1:]
# # print(list(itertools.combinations([[1,2,3],[10,20,30], [100,200,300]],2)))
#
# pares = itertools.combinations([data[my_columns[0]], data[my_columns[1]], data[my_columns[2]], data[my_columns[3]]], 2)
# names_pares = list(itertools.combinations(my_columns,2))
# # print(names_pares)
#
# p_values = []
# for i,pare in enumerate(pares):
#     print('Пара методов: ', names_pares[i])
#     print('Критерий знаковых рангов Вилкоксона:')
#     print('\t',stats.wilcoxon(pare[0], pare[1]))
#     p_value = stats.wilcoxon(pare[0], pare[1])[1]
#     p_values.append(p_value)
#
# print(p_values)
# answer43 = len(list(filter(lambda x: x<0.05, p_values)))
# print('answer 4.3. = ', answer43)
#
# print('4.5. Количество статистически значимых на уровне 0.05 различий с учётом поправки Холма:')
# reject, p_corrected, a1, a2 = multipletests(p_values, alpha=0.05, method='holm')
# answer45 = len(list(filter(lambda x: x<0.05, p_corrected)))
# print('answer 4.5. = ', answer45)
#
# print('4.6. Количество статистически значимых на уровне 0.05 различий с учётом поправки Бенджамини-Хохберга:')
# reject, p_corrected, a1, a2 = multipletests(p_values, alpha=0.05, method='fdr_bh')
# answer46 = len(list(filter(lambda x: x<0.05, p_corrected)))
# print('answer 4.6. = ', answer46)

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)

data = pd.read_csv('botswana.tsv', sep='\t', header=0)
print(data.head())
print(data.describe())

print('Сколько разных значений принимает признак "religion"?')
print(data['religion'].value_counts())
answer61 = len(data['religion'].value_counts())
print('answer 6.1. = ', answer61)

print('6.2. Сколько объектов из 4361 останется, если выбросить все, содержащие пропуски?')




res = (data[(data['heduc'].isnull() == 0) & (data['agefm'].isnull() == 0) & (data['idlnchld'].isnull() == 0) &
           (data['knowmeth'].isnull() == 0) & (data['usemeth'].isnull() == 0) & (data['electric'].isnull() == 0) &
      (data['radio'].isnull() == 0) & (data['tv'].isnull() == 0) & (data['bicycle'].isnull() == 0)])
answer62 = res.shape[0]
print('answer 6.2. = ', answer62)

# data['nevermarr'] = 1 - data['evermarr']
data['nevermarr']=0
data.loc[data['agefm'].isnull(), 'nevermarr']=1
data.drop('evermarr', axis=1, inplace=True)

data.loc[data['agefm'].isnull(), 'agefm']=0
data.loc[data['heduc'].isnull() & data['nevermarr']==1, 'heduc']=-1
print(data.head())

print('6.3. Сколько осталось пропущенных значений в признаке "heduc"?')
answer63 = data[data['heduc'].isnull()].shape[0]
print('answer 6.3. = ', answer63)

print('6.4. Какого размера теперь наша матрица данных? Умножьте количество строк на количество всех столбцов:')
my_columns = ['idlnchld', 'heduc', 'usemeth']
my_values = [-1, -2, -1]
for i, column in enumerate(my_columns):
    new_column = column + '_noans'
    data[new_column] = 0
    data.loc[data[column].isnull(), new_column] = 1
    data.loc[data[column].isnull(), column] = my_values[i]

my_columns = ['knowmeth', 'electric', 'radio', 'tv', 'bicycle']
for column in my_columns:
    data.drop(data.loc[data[column].isnull()].index, axis=0, inplace=True)

print(data.head())
answer64 = data.shape[0]*data.shape[1]
print('answer 6.4. = ', answer64)

print('6.5. Какого размера теперь наша матрица данных? Умножьте количество строк на количество всех столбцов:')
m1 = smf.ols('ceb ~ age+educ+religion+idlnchld+knowmeth+usemeth+agefm+heduc+urban+electric+radio+tv+bicycle+nevermarr+idlnchld_noans+heduc_noans+usemeth_noans', data=data)
fitted = m1.fit()
res = fitted.summary()
print(res, '\n')
answer65 = round(float(str(list(res.tables[0][0])[3]).strip()), 3)
print('answer 6.5. = ', answer65)

print('Проверка гомоскедастичности ошибки по критерию Бройша-Пагана:')
print('p = ', sms.het_breuschpagan(fitted.resid, fitted.model.exog)[1])

m2 = smf.ols('ceb ~ age+educ+religion+idlnchld+knowmeth+usemeth+agefm+heduc+urban+electric+radio+tv+bicycle+nevermarr+idlnchld_noans+heduc_noans+usemeth_noans', data=data)
fitted = m2.fit(cov_type='HC1')
print(fitted.summary())

m3 = smf.ols('ceb ~ age+educ+idlnchld+knowmeth+usemeth+agefm+heduc+urban+electric+bicycle+nevermarr+idlnchld_noans+heduc_noans+usemeth_noans', data=data)
fitted = m3.fit(cov_type='HC1')
print(fitted.summary())

print('6.8. Проверьте с помощью критерия Фишера. Чему равен его достигаемый уровень значимости?')
answer68 = round(m2.fit().compare_f_test(m3.fit())[1],4)
print('answer 6.8. = ', answer68)

print('6.9. Проверьте критерием Фишера гипотезу о том, что качество модели не ухудшилось:')

m4 = smf.ols('ceb ~ age+educ+idlnchld+knowmeth+agefm+heduc+urban+electric+bicycle+nevermarr+idlnchld_noans+heduc_noans', data=data)
fitted = m4.fit(cov_type='HC1')
print(m3.fit().compare_f_test(m4.fit())[1])
answer69 = 40
print('answer 6.9. = ', answer69)

print(fitted.summary())
