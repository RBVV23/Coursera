import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def calculate_means(numeric_data):
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull().values)[0] # - авторская версия
        correction = np.amax(to_sum[indices])
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
    return pd.Series(means, numeric_data.columns)


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 100)


data = pd.read_csv('data.csv')
print(data.shape)
