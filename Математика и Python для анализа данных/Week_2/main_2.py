# Знакомство с библиотеками Pandas, NumPy, SciPy, Matplotlib

import pandas as pd
import numpy as np
from scipy import optimize
from scipy import interpolate
from scipy import linalg
import matplotlib.pyplot as plt


# print(pd.__version__)
print(np.__version__)

# frame = pd.DataFrame( {'numbers': range(10), 'chars': ['a']*10} )
# print(frame)

# frame = pd.read_csv('dataset.tsv')
# print(frame)
frame = pd.read_csv('dataset.tsv', header=0, sep='\t')
print(frame)
print(frame.columns)
print(frame.shape)

new_line = {'Name': 'Perov', 'Birth': '22.03.1990', 'City': 'Penza'}

# frame.append(new_line, ignore_index=True)
frame = frame.append(new_line, ignore_index=True)
print(frame)

frame['IsStudent'] = [False]*5 + [True]*2
print(frame)
# frame = frame.drop([6,5], axis=0)
frame.drop([6, 5], axis=0, inplace=True)
print(frame)
frame.drop('IsStudent', axis=1, inplace=True)
print(frame)

frame.to_csv('dataset_upd.csv', sep=',', header=True, index=None)

frame = pd.read_csv('dataset_upd.csv', header=0, sep=',')
print(frame)
print(frame.dtypes)

frame.Birth = frame.Birth.apply(pd.to_datetime)
print(frame)
print(frame.dtypes)

frame.info()
frame.fillna('разнорабочий', inplace=True)

print(frame['Position'])

# print(frame.Position)
## print(type(frame))

print(frame[['Name', 'Position']])

# print(frame.head(3))
# print(frame[-3:])

# print(frame.loc[[1, 3, 4], ['Name', 'City']])
print(frame.iloc[[1, 3, 4], [0, 2]])

# print(frame[frame.Birth >= pd.datetime(1985,1,1)])
# print(frame[frame['Birth'] >= pd.datetime(1985,1,1)])


# print(frame[(frame['Birth'] >= pd.datetime(1985,1,1)) & (frame.City != 'Москва')])
# print(frame[(frame.Birth >= pd.datetime(1985,1,1)) | (frame['City'] == 'Волгоград')])

###

x = [2, 3, 4, 6]
y = np.array(x)

print(y[[0,2]])
print([y>3])
print(y[y>3])
print(y*5)


matrix = [[1, 2, 4], [3,1,0]]
nd_array = np.array(matrix)

print(matrix[1][2])
print(nd_array[1][2])
print(nd_array[1,2])

print(np.random.rand(6))
print(np.random.randn(4, 5))

print(np.arange(0,8,0.1))

def f(x):
    return (x[0] - 3.2)**2 + (x[1] - 0.1)**2 + 3

## print(f([0, 0]))
print(f([3.2, 0.1]))

x_min = optimize.minimize(f, [0, 0])
print(x_min)
## print(type(x_min))
print(x_min.x)

a = np.array([[3,2,0], [1, -1, 0], [0,5,1]])
b = np.array([2, 4, -1])

x = linalg.solve(a, b)
print(x)

print(np.dot(a, x))
print(b - np.dot(a, x))

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()

x = np.arange(-10, 10, 0.1)
y = x ** 3

plt.plot(x, y)
plt.show()

x = np.arange(0, 10, 2)
# y = np.exp(-x/3.0)
y = np.exp(-x/3.0) + np.random.randn(len(x)) * 0.1

print(x[:5])
print(y[:5])

# f = interpolate.interp1d(x, y, kind='linear')
f = interpolate.interp1d(x, y, kind='quadratic')
xnew = np.arange(0, 8, 0.1)
ynew = f(xnew)

# print(type(f))

plt.plot(xnew, ynew, '-', x, y, 'o')
plt.show()

def f(x):
    return .5*(1 - x[0])**2 + ((x[1] - x[0]**2))**2

print(f([1, 1]))

## x = np.arange(-5, 5, 0.5)
## y = x**2
#
## plt.plot(x, y)
## plt.show()

# result = optimize.brute(f, ((-5, 5), (-5, 5)))
# print(result)
#
# result = optimize.differential_evolution(f, ((-5, 5), (-5, 5)))
# print(result)

def g(x):
    return np.array((-2*.5*(1-x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2  )))

print(optimize.check_grad(f, g, [2, 2]))
##print(g([3, 4]))

# print(optimize.fmin_bfgs(f, [2,2], fprime=g))

print(optimize.minimize(f, [2,2]))
# print(optimize.minimize(f, [2,2], method='BFGS'))
print(optimize.minimize(f, [2,2], method='BFGS', jac=g))


d = np.array([3, 0, 8, 9, -10])
print('Вектор d:', d)
print('Его размерность:', d.shape)

print ('Вектор d с newaxis --> вектор-строка:\n', d[np.newaxis, :])
print ('Полученная размерность:', d[np.newaxis, :].shape)
print ('Вектор d с newaxis --> вектор-столбец:\n', d[:, np.newaxis])
print ('Полученная размерность:', d[:, np.newaxis].shape)


a = np.array([[1, 2, 3], [4,5,6], [7,8,9]])
print(a)
b = np.eye(4)
print(b)
c = np.ones((2, 6))
print(c)
d = np.zeros((6,2))
print(d)
v = np.arange(0.5, 9, 1.5)
print(v)
w = v.reshape((2,3))
print(w)

# e = np.arange(1, 13, 2).reshape((3, 2))

a = np.array([[1, 0], [0, 1]])
b = np.array([[4, 1], [2, 2]])
r1 = np.dot(a, b)
r2 = a.dot(b)

print("Матрица A:\n", a)
print("Матрица B:\n", b)
print("Результат умножения функцией:\n", r1)
print("Результат умножения методом:\n", r2)

a = np.array([[1, 2, 1], [1, 1, 4], [2, 3, 6]])
det = np.linalg.det(a)
print(det)

a = np.array([[1, 2, 3], [1, 1, 1], [2, 2, 2]])
r = np.linalg.matrix_rank(a)
print(r)

a = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(a, b)
print("Матрица A:\n", a)
print("Вектор b:\n", b)
print("Решение системы:\n", x)
print(np.dot(a,x)-b)

a = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
b = np.array([-1, 0.2, 0.9, 2.1])
x, res, r, s = np.linalg.lstsq(a, b, rcond=-1)
print("Псевдорешение системы:\n", x)

a = np.array([[1, 2, 1], [1, 1, 4], [2, 3, 6]])
b = np.linalg.inv(a)
print("Матрица A:\n", a)
print("Обратная матрица к A:\n", b)
print("Произведение A на обратную должна быть единичной:\n", a.dot(b))

a = np.array([[-1, -6], [2, 6]])
w, v = np.linalg.eig(a)
print("Матрица A:\n", a)
print("Собственные числа:\n", w)
print("Собственные векторы:\n", v)

a = 3 + 2j
b = 1j
print("Комплексное число a:\n", a)
print("Комплексное число b:\n", b)

c = a * a
e = b*b
d = a / (4 - 5j)
print("Комплексное число c:\n", c)
print("Комплексное число e:\n", e)
print("Комплексное число d:\n", d)