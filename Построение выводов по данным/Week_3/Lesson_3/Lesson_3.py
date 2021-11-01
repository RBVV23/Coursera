import patsy
# print(patsy.__version__)
import statsmodels
import scipy as sc
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.graphics.regressionplots import plot_leverage_resid2
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
raw = pd.read_csv('beauty.csv', sep=';', index_col=False)
print(raw.head(), '\n')

pd.plotting.scatter_matrix(raw[['wage', 'exper', 'educ', 'looks']], alpha=0.2, figsize=(15,15), diagonal='hist')
# plt.show()

cat_features = ['union', 'goodhlth', 'black', 'female', 'married', 'service']
for cat in cat_features:
    print(raw[cat].value_counts())

data = raw

plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
data['wage'].plot.hist()
plt.xlabel('Wage', fontsize=14)

plt.subplot(1,2,2)
np.log(data['wage']).plot.hist()
plt.xlabel('Log wage', fontsize=14)
# plt.show()

data = data[data['wage'] < 77]

plt.figure(figsize=(8,7))
data.groupby('looks')['looks'].agg(lambda x: len(x)).plot(kind='bar', width=0.9)
plt.xticks(rotation=0)
plt.xlabel('Looks', fontsize=14)
# plt.show()

data['belowavg'] = data['looks'].apply(lambda x: 1 if x < 3 else 0)
data['aboveavg'] = data['looks'].apply(lambda x: 1 if x > 3 else 0)
data.drop('looks', axis=1, inplace=True)

print(data.head())

m1 = smf.ols('wage ~ exper + union + goodhlth + black + female + married + service + educ + belowavg + aboveavg', data=data)
fitted = m1.fit()
print(fitted.summary())

plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
sc.stats.probplot(fitted.resid, dist='norm', plot=plt)
plt.subplot(1,2,2)
np.log(fitted.resid).plot.hist()
plt.xlabel('Residuals', fontsize=14)
# plt.show()

m2 = smf.ols('np.log(wage) ~ exper + union + goodhlth + black + female + married + service + educ + belowavg + aboveavg', data=data)
fitted = m2.fit()
print(fitted.summary())

plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
sc.stats.probplot(fitted.resid, dist='norm', plot=plt)
plt.subplot(1,2,2)
np.log(fitted.resid).plot.hist()
plt.xlabel('Residuals', fontsize=14)
# plt.show()

plt.figure(figsize=(16,7))
plt.subplot(1,2,1)

plt.subplot(1,2,2)
plt.scatter(data['educ'], fitted.resid)
plt.xlabel('Educations', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()