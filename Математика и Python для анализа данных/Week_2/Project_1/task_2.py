import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def f(x):

    return np.sin(x/5)*np.exp(x/10) + 5*np.exp(-x/2)

x = np.arange(0, 15.05, 0.05)
# print(x)
y = f(x)
# print(y)

# plt.plot(x, y)
# plt.show()

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

b = np.array([y_sol[0], y_sol[1], y_sol[2]])
W=np.array(linalg.solve(A, b))
print('W:')
print(W[0], W[1], W[2])

def f2_pol(inp, W_matr):
    w_0 = W_matr[0]
    w_1 = W_matr[1]
    w_2 = W_matr[2]
    return w_2*inp*inp + w_1*inp + w_0

y_2 = f2_pol(x, W)
plt.plot(x, y, x, y_2, x_sol, y_sol, 'o')
plt.show()

