import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from scipy import stats


pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 100)


data = pd.read_csv('churn_analysis.csv')
print(data.head())
print(data.info())

print(np.unique(data.state))
print(len(np.unique(data.state)))

treatment = data[data.treatment == 1]
all_states = np.unique(data.state)
L = len(all_states)
n = 0
p_values = []
n_corrected = 0
p_values_corrected = []
n_fisher = 0
p_values_fisher = []
f1, f2, f3 = 0,0,0
g1, g2, g3 = 0, 0, 0

for n1, st1 in enumerate(all_states):
    for n2 in range(n1+1,L):
        st2 = all_states[n2]
        table = np.zeros((2,2))
        table[0,0] = treatment[(treatment.state == st1) & (treatment.churn == 'False.')].shape[0]
        table[0,1] = treatment[(treatment.state == st1) & (treatment.churn == 'True.')].shape[0]
        table[1,0] = treatment[(treatment.state == st2) & (treatment.churn == 'False.')].shape[0]
        table[1,1] = treatment[(treatment.state == st2) & (treatment.churn == 'True.')].shape[0]
        p_value = chi2_contingency(table, correction=False)[1]
        p_values.append(p_value)
        p_value_corrected = chi2_contingency(table, correction=True)[1]
        p_values_corrected.append(p_value_corrected)
        p_value_fisher = fisher_exact(table)[1]
        p_values_fisher.append(p_value_fisher)
        if p_value < 0.05:
            n += 1
        if p_value_corrected < 0.05:
            n_corrected += 1
        if p_value_fisher < 0.05:
            n_fisher += 1
        if (p_value > p_value_corrected) and (f1 == 0):
            print('УМЕНЬШЕНИЕ значения p-value после поправки Йетса:')
            print('\tp-value = {} -> p-value (с поправкой Йетса): {}'.format(p_value, p_value_corrected))
            f1 = 1
        if (p_value < p_value_corrected) and (f2 == 0):
            print('УВЕЛИЧЕНИЕ значения p-value после поправки Йетса:')
            print('\tp-value = {} -> p-value (с поправкой Йетса): {}'.format(p_value, p_value_corrected))
            f2 = 1
        if (p_value == p_value_corrected) and (f3 == 0):
            print('СОХРАНЕНИЕ p-value после поправки Йетса:')
            print('\tp-value = {} -> p-value (с поправкой Йетса): {}'.format(p_value, p_value_corrected))
            f3 = 1
        if (p_value > p_value_fisher) and (g1 == 0):
            print('УМЕНЬШЕНИЕ значения p-value при использовании точного критерия Фишера:')
            print('\tp-value = {} -> p-value (точный критерий Фишера): {}'.format(p_value, p_value_fisher))
            g1 = 1
        if (p_value < p_value_fisher) and (g2 == 0):
            print('УВЕЛИЧЕНИЕ значения p-value при использовании точного критерия Фишера:')
            print('\tp-value = {} -> p-value (точный критерий Фишера): {}'.format(p_value, p_value_fisher))
            g2 = 1
        if (p_value == p_value_fisher) and (g3 == 0):
            print('СОХРАНЕНИЕ p-value при использовании точного критерия Фишера:')
            print('\tp-value = {} -> p-value (точный критерий Фишера): {}'.format(p_value, p_value_fisher))
            g3 = 1
answer1 = n
print()
print('answer1 = ', answer1)
print('Количество отвергнутых нулевых гипотез (p-value < 0.05) с учетом поправки Йетса:', n_corrected)
print('Количество отвергнутых нулевых гипотез (p-value < 0.05) при использовании точного критерия Фишера:', n_corrected)
print()
print('Среденее значение достигаемого уровня значимости до введения каких-либо поправок: ', np.mean(p_values))
print('Среденее значение достигаемого уровня значимости после введения поправки Йетса: ', np.mean(p_values_corrected))
print('Среденее значение достигаемого уровня значимости при использовании точного критерия Фишера: ', np.mean(p_values_fisher))


corr_p = data.corr(method='pearson')['day_calls']['mes_estim']
print()
print('Корреляция Пирсона между признаками "day_calls" и  "mes_estim":', corr_p)
n = data.shape[0]
T = corr_p*np.sqrt((n-2))/np.sqrt(1 - corr_p**2)
p_value = stats.t.cdf(T, n-2)
print('\tДостигаемый уровень значимости:', p_value)

corr_s = data.corr(method='spearman')['day_calls']['mes_estim']
print('Корреляция Спирмена между признаками "day_calls" и  "mes_estim":', corr_s)
n = data.shape[0]
T = corr_s*np.sqrt((n-2))/np.sqrt(1 - corr_s**2)
p_value = stats.t.cdf(T, n-2)
print('\tДостигаемый уровень значимости:', p_value)

treatment = data[data.treatment == 1]