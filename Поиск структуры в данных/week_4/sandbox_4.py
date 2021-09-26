import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym

def my_polinom_gen(degree=3, int=True):
    a = np.random.randint(-10,10,degree)
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
# print(sym.factor(g))
# print(np.random.randint(0,3,2))
print(my_polinom_gen(3))


