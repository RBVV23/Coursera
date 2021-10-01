import numpy as np
import sympy as sym


# factor - разложить на множители
# expand - раскрыть скобки
# simplfy

def my_polinom_gen(degree=3, maxabs=10, answer=True, int=True):
    a = np.random.randint(-maxabs,maxabs,degree)
    sym.var('x')
    f = 1
    for i in range(degree):
        f *= (x-a[i])
    if answer:
        print(a)
    return sym.expand(f)
sym.var('a')
sym.var('b')
sym.var('x')

print(sym.expand(f))
print(sym.simplify(g))
print(sym.factor(x**3 - 6*x**2 - 67*x + 360))
print(np.random.randint(0,3,2))
print(my_polinom_gen(3))
g = sym.factor(x**3 - 6*x**2 - 67*x + 360)
print(str(g)[:5])

print(sym.expand((2*x+1)**3))
print(sym.expand((x+2)**3))


