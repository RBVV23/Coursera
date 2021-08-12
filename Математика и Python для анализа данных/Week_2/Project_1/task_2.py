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

# linalg.solve(x_sol, y_sol)

