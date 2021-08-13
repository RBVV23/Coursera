import numpy as np
np.__version__
import pandas as pd
pd.__version__
import matplotlib.pyplot as plt
from scipy import optimize
import math as m
# m.__version__



def h(x):
    return int(m.sin(x/5)*m.exp(x/10) + 5*m.exp(-x/2))

print(type(h(5)))



def report(start_x, results):
    print('\n=================== Отчёт ===================\n')
    print('\tНачальная точка: ' + str(start_x))
    print('\tТочка минимума: ' + str(results['x']))
    print('\tЗначение минимума: ' + str(results['fun']))
    print('\tКоличество итераций: ' + str(results['nit']))
    print('\n=============== Конец отчёта ===============\n')
    return


x = []
y = []
ans=0
dx = 0.05
while ans <= 30:
    ans += dx
    ans = round(ans, 2)
    x.append(ans)
    ans_2 = h(ans)
    y.append(ans_2)

# print(x)

# y = h(x)
plt.plot(x, y)
plt.show()


# метод Бройдена — Флетчера — Гольдфарба — Шанно (BFGS), начальное приближение x=30
res = optimize.minimize(h, 30, method='BFGS')
report(30, res)

# метод дифференциальной эволюции
res = optimize.differential_evolution(h, [(1, 30)])
report(None, res)
