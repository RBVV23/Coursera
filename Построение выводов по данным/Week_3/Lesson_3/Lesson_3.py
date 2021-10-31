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

raw = pd.read_csv('beauty.csv', sep=';', index_col=False)
print(raw.head(), '\n')

pd.plotting.scatter_matrix(raw[['wage', 'exper', 'educ', 'looks']], alpha=0.2, figsize=(15,15), diagonal='hist')
# plt.show()

cat_features = ['union', 'goodhlth', 'black', 'female', 'married', 'service']
for cat in cat_features:
    print(raw[cat].value_counts())