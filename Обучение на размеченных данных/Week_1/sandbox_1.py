import pandas as pd
# pd.__version__
import seaborn as sns
sns.__version__
import numpy as np
np.__version__
import matplotlib.pyplot as plt
# plt.__version__

def make_bmi(height, weight):
    METER_TO_INCH = 39.37
    KILO_TO_POUND = 2.20462
    height_inch = height / METER_TO_INCH
    weight_pound = weight / KILO_TO_POUND
    bmi = weight_pound/(height_inch**2)
    return bmi


# data = pd.read_csv('weights_heights.csv')
data = pd.read_csv('weights_heights.csv',  index_col='Index')

print(data.head())
# print(data.shape)
# print(data.info())

data.plot(y='Height', kind='hist', color='red', title='Распределние роста подростков (дюймы)')

data['BMI'] = make_bmi(data['Height'], data['Weight'])
# print(data.head())

# sns_plot = sns.pairplot(data)
sns_plot = sns.pairplot(data[['Height', 'Weight', 'BMI']])
sns_plot.savefig('pairplot.png')
