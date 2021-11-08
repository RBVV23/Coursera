import scipy
from scipy import stats
import statsmodels.stats.multitest as smm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 72)

data = pd.read_csv('gene_high_throughput_sequencing.csv', header=0, sep=',')
print(data.head())
print(data.shape)

control = 'normal'
test = 'cancer'
gens = data.columns[2:]
print(gens)

gens = ['LOC643837','LOC100130417']
for gen in gens:
    print('gen:', gen)
    values_1 = data[gen][data['Diagnosis'] == control]
    # print(values_1)
    # print(values_2)
    values_2 = data[gen][data['Diagnosis'] == test]
    # p_value = scipy.stats.ttest_ind(values_1, values_2, equal_var=False)
    # print(p_value)
    df = my_t_test_df(values_1, values_2)
    t_stat = my_t_statistic_ind(values_1, values_2)
    print(t_stat, 2*(1 - stats.t.cdf(t_stat, df)))
    p_value = 2*(1 - stats.t.cdf(t_stat, df))
    # if p_value < 0.05:

