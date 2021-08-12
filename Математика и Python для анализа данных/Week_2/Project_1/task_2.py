import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def f(x):

    return np.sin(x/5)*np.exp(x/10) + 5*np.exp(-x/2)

x = np.arange(0, 15.05, 0.05)
# print(x)
y = f(x)
# print(y)

plt.plot(x, y)
plt.show()

x_sol = np.array([1., 15.])
y_sol = f(x_sol)
print(y_sol)

# Ax = b

A = np.array([ [1, x_sol[0]], [1, x_sol[1]] ])
print(A)
b = np.array([y_sol[0], y_sol[1]])
print(b)

W=np.array(linalg.solve(A, b))
print('W:')
print(W[0], W[1])

def f1_pol(inp, W_matr):
    # print('x=' + str(inp))
    w_0 = W_matr[0]
    # print('w_0=' + str(w_0))
    w_1 = W_matr[1]
    # print('w_1=' + str(w_1))
    return w_1*inp + w_0

x = np.arange(0, 15.05, 0.05)
y_1 = f1_pol(x, W)
# print(x)
# f_pol(1, W)
# print(f_pol(1, W))
# f_pol(15, W)
# print(f_pol(15, W))

plt.plot(x, y, x, y_1, x_sol, y_sol, 'o')
plt.show()

x_sol = np.array([1.,8., 15.])
y_sol = f(x_sol)

A = np.array([ [1, x_sol[0], x_sol[0]**2],
               [1, x_sol[1], x_sol[1]**2],
               [1, x_sol[2], x_sol[2]**2] ])

b = np.array(y_sol)
W=np.array(linalg.solve(A, b))
print('W:')
print(W[0], W[1], W[2])

def f_pol(inp, W_matr):
    res = 0
    w = []
    print()
    for i in range(len(W_matr)):
        w.append(W_matr[i])
        res += w[i]*inp**(i)
        print(str(w[i]) + ' * x^' + str(i))
    return res

y_2 = f_pol(x, W)
plt.plot(x, y, x, y_2, x_sol, y_sol, 'o')
plt.show()

x_sol = np.array([1.,4., 10., 15.])
y_sol = f(x_sol)

myA = []
for var in x_sol:
    A_str = []
    for n in range(len(x_sol)):
        A_str.append(var**n)
    myA.append(A_str)
my_A = np.array(myA)


b = np.array(y_sol)
W=np.array(linalg.solve(my_A, b))

y_3 = f_pol(x, W)
plt.plot(x, y, x, y_3, x_sol, y_sol, 'o')
plt.show()

print('Коэффициенты для аппроксимации (до сотых):\n\t', np.round(W, 2))