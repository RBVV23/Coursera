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

def weight_category(weight):
    pass
    cat = 2
    if weight < 120:
        cat = 1
    elif weight >= 150:
        cat = 3
    return cat


# data = pd.read_csv('weights_heights.csv')
data = pd.read_csv('weights_heights.csv',  index_col='Index')

print(data.head())
# print(data.shape)
# print(data.info())

## data.plot(y='Height', kind='hist', color='red', title='Распределние роста подростков (дюймы)')

data['BMI'] = make_bmi(data['Height'], data['Weight']) # Саня, здесь я применил свою функцию к столбцам "в лоб" и питон сам все сделал
# data['weight_cat'] = weight_category(data['Weight']) # Почему тогда здесь не прокатил аналогичный способ?

data['weight_cat'] = data['Weight'].apply(weight_category) # здесь я использовал apply, чтобы применить функцию к целому столбцу
# и можно было функцию make_bmi тож сделать через apply?

print(data.head())

# sns_plot = sns.pairplot(data)
## sns_plot = sns.pairplot(data[['Height', 'Weight', 'BMI']])
# sns_plot.savefig('pairplot.png')
## plt.show()


## boxplot = sns.boxplot(y='Height', x='weight_cat', data=data, orient='v')
## boxplot.set_xlabel('Весовая категория')
## boxplot.set_ylabel('Рост')
## plt.show()

## data.plot(x='Weight', y='Height', kind='scatter')
## plt.show()

Y = data['Height']
print(len(Y))
# print(Y[1])
X = np.linspace(0, 200, 25000)
print('START')

def my_error(w0, w1):
    sum = 0
    for i in range(1, len(Y)):
        er = (Y[i] - (w0 + w1*X[i]))**2
        sum += er
    return sum
print('pre', my_error(60, 0.05))

X = np.linspace(0, 200, 25000)
Y = data['Height']
w_0 = 50
w_1 = []
dw = 0.01
Er = []
for i in range(100):
    w_1.append(dw*i)
    # print(w_1[i])
    Er.append(my_error(w_0, w_1[i]))
plt.plot(w_1, Er)
plt.show()
