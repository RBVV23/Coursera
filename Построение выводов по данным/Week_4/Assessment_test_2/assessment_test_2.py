import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, probplot
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from statsmodels.sandbox.stats.multicomp import multipletests

def my_proportions_confint_diff_ind(sample1, sample2, alpha=0.05):
    n1 = len(sample1)
    n2 = len(sample2)
    p1 = float(sum(sample1)/n1)
    p2 = float(sum(sample2)/n2)
    z = stats.norm.ppf(1-alpha/2.)
    low = p1-p2 - z*np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    high = p1-p2 + z*np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    return low, high

def my_v_cramer(table):
    K1 = table.shape[0]
    K2 = table.shape[1]
    N = np.sum(table)
    minK = min(K1, K2)
    chi2 = stats.chi2_contingency(table)[0]
    return np.sqrt(chi2 / (N*(minK-1)))


pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 100)


data = pd.read_csv('churn_analysis.csv')
print(data.head())
print(data.info())

# print(np.unique(data.state))
# print(len(np.unique(data.state)))

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

# print('p_values:')
# print(p_values)
# print('p_values_corrected:')
# print(p_values_corrected)
# print('p_values_fisher:')
# print(p_values_fisher)

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
T = corr_p*np.sqrt(n-2)/np.sqrt(1 - corr_p**2)
# print(T)
p_value = 2*(1 - stats.t.cdf(abs(T), n-2))
print('\tДостигаемый уровень значимости:', p_value)

corr_s = data.corr(method='spearman')['day_calls']['mes_estim']
print('Корреляция Спирмена между признаками "day_calls" и  "mes_estim":', corr_s)
n = data.shape[0]
T = corr_s*np.sqrt((n-2))/np.sqrt(1 - corr_s**2)
p_value = 2*(1 - stats.t.cdf(abs(T), n-2))
print('\tДостигаемый уровень значимости:', p_value)


plt.subplot(1,2,1)
# N = len(control_mean)
probplot(data.day_calls, dist="norm", plot=plt)
# print(R_squared)
plt.subplot(1,2,2)
# N = len(control_boot_chi_squared)
probplot(data.mes_estim, dist="norm", plot=plt)
# plt.show()

plt.figure()
plt.scatter(data.day_calls, data.mes_estim, alpha=0.2)
plt.xlabel('day_calls')
plt.ylabel('mes_estim')
# plt.show()

new_data = data[data.treatment == 1]
table = np.zeros((len(np.unique(new_data.state)), len(np.unique(new_data.churn))))
# print(np.unique(new_data.state))
# print(np.unique(new_data.churn))
for x, ch in enumerate(np.unique(new_data.churn)):
    for y, st in enumerate(np.unique(new_data.state)):
        # print('ch = {}; st = {}'.format(ch,st))
        cell = new_data[(new_data.churn == ch) & (new_data.state == st)]
        # print(cell)
        # print(cell.shape[0])
        table[y, x] = cell.shape[0]

print('Построим таблицу сопряженности:')
print(table)
corr_cr = my_v_cramer(table)
print('Коэффициент V Крамера: ', round(my_v_cramer(table),4))
n = new_data.shape[0]
T = stats.chi2_contingency(table)[0]
df = (table[0] - 1)*(table[1] - 1)
# p_value = 2*(1 - stats.chi2.cdf(abs(T), df))
p_value = stats.chi2_contingency(table)[1]
print('Достигаемый уровень значимости: ', p_value)

new_table = pd.pivot_table(data, values='account_length', index='state', columns=['treatment', 'churn'], aggfunc=len, fill_value=0)
print(new_table)


for i in range(3):
    sums = data[(data.treatment == i) & (data.churn == 'True.')].shape[0]
    all = data[(data.treatment == i)].shape[0]
    prop = sums/all
    print('treatment = ', i)
    print('proportion = {}'.format(round(prop,4)))
    interval = np.round(proportion_confint(count=sums, nobs=all,
                                           method='wilson'), 4)
    print('[{}; {}]'.format(interval[0], interval[1]))
# prop_1 = data[(data.treatment == 1) & (data.churn == 'True.')].shape[0] / data[(data.treatment == 1)].shape[0]
# prop_2 = data[(data.treatment == 2) & (data.churn == 'True.')].shape[0] / data[(data.treatment == 2)].shape[0]

p_values = []
for i in range(0,3):
    for j in range(i+1,3):
        s1 = data[(data.treatment == i) & (data.churn == 'True.')].shape[0]
        l1 = data[(data.treatment == i)].shape[0]
        s2 = data[(data.treatment == j) & (data.churn == 'True.')].shape[0]
        l2 = data[(data.treatment == j)].shape[0]
        p1 = float(s1/l1)
        p2 = float(s2/l2)
        z = stats.norm.ppf(1 - 0.05 / 2.)
        low = p1 - p2 - z * np.sqrt(p1 * (1 - p1) / l1 + p2 * (1 - p2) / l2)
        high = p1 - p2 + z * np.sqrt(p1 * (1 - p1) / l1 + p2 * (1 - p2) / l2)
        P = (p1 * l1 + p2 * l2) / (l1 + l2)
        z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1 / l1 + 1 / l2))
        p_value = 2*(1 - stats.norm.cdf(abs(z_stat)))
        print('[treatment = {}] - [treatment = {}]'.format(i, j))
        print('\t[{}; {}]'.format(round(low,4), round(high,4)))
        print('\tp-value = ', round(p_value,4))
        p_values.append(p_value)

_, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print(p_corrected)