import numpy as np
np.__version__
import pandas as pd
pd.__version__
import matplotlib.pyplot as plt
from scipy import optimize


def f(x):
    return np.sin(x/5)*np.exp(x/10) + 5*np.exp(-x/2)

def report(results):
    print('\n=================== Отчёт ===================\n')
    print('\tТочка минимума: ' + str(results['x']))
    print('\tЗначение минимума: ' + str(results['fun']))
    print('\tКоличество итераций: ' + str(results['nit']))
    print('\n=============== Конец отчёта ===============\n')
    return

x = np.arange(1, 30.05, 0.05)
# print(x)
y = f(x)
# plt.plot(x, y)
# plt.show()

# res = optimize.minimize(f, x_start[i])

res = optimize.differential_evolution(f, [(1, 30)])
report(res)