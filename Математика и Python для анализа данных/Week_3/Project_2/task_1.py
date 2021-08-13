import numpy as np
np.__version__
import pandas as pd
pd.__version__
import matplotlib.pyplot as plt
from scipy import optimize

def f(x):
    return np.sin(x/5)*np.exp(x/10) + 5*np.exp(-x/2)

def report(start_x, results):
    print('\n=================== Отчёт ===================\n')
    print('\tНачальная точка: ' + str(start_x))
    print('\tТочка минимума: ' + str(results['x']))
    print('\tЗначение минимума: ' + str(results['fun']))
    print('\tКоличество итераций: ' + str(results['nit']))
    print('\n=============== Конец отчёта ===============\n')
    return

x = np.arange(1, 30.05, 0.05)
# print(x)
y = f(x)
plt.plot(x, y)
plt.show()



# параметры по умолчанию
# x_start = 1 + 30*np.random.rand(10)
# x_start = np.round(x_start, 1)
x_start = np.array([3.8, 17.2])
# print(x_start)

for i in range(len(x_start)):
    res = optimize.minimize(f, x_start[i])
    report(x_start[i], res)

# метод Бройдена — Флетчера — Гольдфарба — Шанно (BFGS), начальное приближение x=2
res = optimize.minimize(f, 2, method='BFGS')
# print(res)
report(2, res)

# метод Бройдена — Флетчера — Гольдфарба — Шанно (BFGS), начальное приближение x=30
res = optimize.minimize(f, 30, method='BFGS')
# print(res)
report(30, res)