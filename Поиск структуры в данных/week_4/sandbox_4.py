import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym

def my_polinom_gen(degree=3, maxabs=10, int=True):
    a = np.random.randint(-maxabs,maxabs,degree)
    sym.var('x')
    f = 1
    for i in range(degree):
        f *= (x-a[i])
    return sym.expand(f)
sym.var('a')
sym.var('b')
sym.var('x')

# print(sym.expand(f))
# print(sym.simplify(g))
print(sym.factor(x**3 - 6*x**2 - 67*x + 360))
# print(np.random.randint(0,3,2))
print(my_polinom_gen(3))


