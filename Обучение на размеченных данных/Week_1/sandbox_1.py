import pandas as pd
# pd.__version__
import seaborn as sns
sns.__version__
import numpy as np
np.__version__
import matplotlib.pyplot as plt
# plt.__version__
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

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

def my_error(w0, w1):
    sum = 0
    for i in range(1, len(Y)):
        er = (Y[i] - (w0 + w1*X[i]))**2
        sum += er
    return sum

# data = pd.read_csv('weights_heights.csv')
data = pd.read_csv('weights_heights.csv',  index_col='Index')

print(data.head())
# print(data.shape)
# print(data.info())

data.plot(y='Height', kind='hist', color='red', title='Распределние роста подростков (дюймы)')

data['BMI'] = make_bmi(data['Height'], data['Weight']) # Саня, здесь я применил свою функцию к столбцам "в лоб" и питон сам все сделал
# data['weight_cat'] = weight_category(data['Weight']) # Почему тогда здесь не прокатил аналогичный способ?

data['weight_cat'] = data['Weight'].apply(weight_category) # здесь я использовал apply, чтобы применить функцию к целому столбцу
# и можно было функцию make_bmi тож сделать через apply?

print(data.head())

# sns_plot = sns.pairplot(data)
sns_plot = sns.pairplot(data[['Height', 'Weight', 'BMI']])
# sns_plot.savefig('pairplot.png')
plt.show()


boxplot = sns.boxplot(y='Height', x='weight_cat', data=data, orient='v')
boxplot.set_xlabel('Весовая категория')
boxplot.set_ylabel('Рост')
plt.show()

data.plot(x='Weight', y='Height', kind='scatter')
plt.show()

Y = data['Height']
print(len(Y))
# print(Y[1])
X = data['Weight']



Y = data['Height']
w_0 = 50
w_1 = []
dw = 0.05
Er = []
for i in range(201):
    w_1.append(-5 + dw*i)
    # print(w_1[i])
    Er.append(my_error(w_0, w_1[i]))
plt.plot(w_1, Er)
plt.grid()
plt.show()


def f(x):
    return my_error(50, x)

opt = optimize.minimize_scalar(f, [-5,5])
print(opt)
w_opt = opt['x']

X = np.linspace(0, 200, 1000)
w_0 = 50
w_1 = w_opt
Y = w_0 + w_1*X
data.plot(x='Weight', y='Height', kind='scatter')
plt.plot(X, Y)
plt.grid()
plt.show()

fig = plt.figure
ax = plt.gca(projection='3d')
W_0 = np.arange(-100, 100, 1)
W_1 = np.arange(-5, 5, 0.01)

W_0, W_1 = np.meshgrid(W_0, W_1)
E = my_error(W_0, W_1)

surf = ax.plot_surface(W_0, W_1, E)
ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Error')
plt.show()


X = data['Weight']
Y = data['Height']

def my_error_v(W):
    sum = 0
    for i in range(1, len(Y)):
        er = (Y[i] - (W[0] + W[1]*X[i]))**2
        sum += er
    return sum

x0 = np.array([0, 1])

res = optimize.minimize(my_error_v, x0, method='L-BFGS-B', bounds=[(-100,100), (-5,5)])
W_opt = res['x']
print(res)

X = np.linspace(0, 200, 1000)

Y_opt = W_opt[0] + W_opt[1]*X
data.plot(x='Weight', y='Height', kind='scatter')

plt.plot(X, Y_opt)
plt.show()


X = np.array([[-5, 7], [9, 8]])
y = np.array([[29], [-11]])

A = np.array([[2, 4, 0],
              [-2, 1, 3],
              [-1, 0, 1]])
print(A.transpose())
Bv = np.array([1, 2, -1])
print(np.dot(A,Bv))