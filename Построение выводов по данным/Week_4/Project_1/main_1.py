import scipy
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests

def my_t_statistic_ind(sample1, sample2):
    M1 = np.mean(sample1)
    M2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    D1 = np.sum((sample1 - M1)**2) / (n1-1)  # несмещенная оценка (ddof = 1)
    D2 = np.sum((sample2 - M2)**2) / (n2 - 1)  # несмещенная оценка (ddof = 1)
    T_stat = (M1 - M2)/np.sqrt(D1/n1 + D2/n2)
    return T_stat
def my_t_test_df(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    D1 = np.sum((sample1 - sample1.mean())**2) / (n1-1)  # несмещенная оценка (ddof = 1)
    D2 = np.sum((sample2 - sample2.mean())**2) / (n2 - 1)  # несмещенная оценка (ddof = 1)
    df = ( (D1/n1 + D2/n2)**2 ) / ( D1**2/((n1-1)*n1**2) + D2**2/((n2-1)*n2**2) )
    return df
def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))
def my_fold_change(T,C, check=False):
    if T > C:
        res = float(T) / C
    if T < C:
        res = -float(C) / T
    if check:
        flag = False
        if abs(res) >= 1.5:
            flag = True
        return res, flag
    return res


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 72)

data = pd.read_csv('gene_high_throughput_sequencing.csv', header=0, sep=',')
print(data.head())
print(data.shape)


gens = data.columns[2:]

p_values_1 = []
p_values_2 = []
counter = 0
counter_2 = 0
control = 'normal'
treatment = 'early neoplasia'
for gen in gens:
    values_1 = data[gen][data['Diagnosis'] == control]
    values_2 = data[gen][data['Diagnosis'] == treatment]
    # p_value = scipy.stats.ttest_ind(values_1, values_2, equal_var=False)[1]
    df = my_t_test_df(values_1, values_2)
    t_stat = my_t_statistic_ind(values_1, values_2)
    my_cdf = stats.t.cdf(t_stat, df)
    if my_cdf < 0.5:
        p_value = 2*my_cdf
    else:
        p_value = 2 * (1 - abs(stats.t.cdf(t_stat, df)))
    p_values_1.append(p_value)
    T = np.mean(values_1)
    C = np.mean(values_2)
    fold_change = my_fold_change(T, C)
    if p_value < 0.05 and abs(fold_change) > 1.5:
        counter_2 += 1
    elif p_value < 0.05:
        counter += 1

answer11 = counter
print('answer11 = ', answer11)
my_write_answer(answer11, part=1, number=1)

control = 'early neoplasia'
treatment = 'cancer'
counter = 0
for gen in gens:
    values_1 = data[gen][data['Diagnosis'] == control]
    values_2 = data[gen][data['Diagnosis'] == treatment]
    df = my_t_test_df(values_1, values_2)
    t_stat = my_t_statistic_ind(values_1, values_2)
    my_cdf = stats.t.cdf(t_stat, df)
    if my_cdf < 0.5:
        p_value = 2*my_cdf
    else:
        p_value = 2 * (1 - abs(stats.t.cdf(t_stat, df)))
    p_values_2.append(p_value)
    if p_value < 0.05:
        counter += 1

answer12 = counter
print('answer12 = ', answer12)
my_write_answer(answer12, part=1, number=2)


reject_1, p_corrected_1, spam, egg = multipletests(p_values_1,
                                            alpha = 0.05,
                                            method = 'holm')
counter = 0
for p in p_corrected_1:
    if p < 0.05/2:
        counter += 1
print('answer21 = ', counter)

reject_2, p_corrected_2, spam, egg = multipletests(p_values_2,
                                            alpha = 0.05,
                                            method = 'holm')
counter = 0
for p in p_corrected_2:
    if p < 0.05/2:
        counter += 1
print('answer22 = ', counter)