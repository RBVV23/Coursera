import numpy as np
np.__version__
import scipy.stats as sts
import matplotlib.pyplot as plt

mn = sts.trapezoid.mean(c=0.15, d=0.35, loc=0, scale=1)
print('Теоретическое матожидание составляет: ', mn)
disp = sts.trapezoid.var(c=0.15, d=0.35, loc=0, scale=1)
print('Теоретическая дисперсия составляет: ', disp)
SKO = disp**(0.5)
print('Теоретическое СКО составляет: ', SKO)

x = np.linspace(0,1)
n = 5
sample_mean = []
for i in range(1000):
    T_n = sts.trapezoid(0.15, 0.35).rvs(n)
#     print(T_n)
    sample_mean.append(np.mean(T_n))
# print(sample_mean)
sample_mean = sample_mean
plt.hist(sample_mean, density=True, stacked = True, label='n=5') #, label='Выборка')
pdf = sts.norm.pdf(x, loc=mn, scale=SKO)
plt.plot(x, pdf, label='Нормальное распределение',  linewidth=3 )
plt.ylabel('Распределение выборочных средних')
plt.xlabel('$x_{cp}$')
plt.legend(loc='upper right')
plt.show()