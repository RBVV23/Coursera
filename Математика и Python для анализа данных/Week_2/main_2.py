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
\
print(x[:5])
print(y[:5])

# f = interpolate.interp1d(x, y, kind='linear')
f = interpolate.interp1d(x, y, kind='quadratic')
xnew = np.arange(0, 8, 0.1)
ynew = f(xnew)

# print(type(f))

plt.plot(xnew, ynew, '-', x, y, 'o')
plt.show()