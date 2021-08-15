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
plt.show()
x = np.linspace(0,1)
n = 5
sample_mean = []
for i in range(1000):
    T_n = sts.trapezoid(0.15, 0.35).rvs(n)
#     print(T_n)
    sample_mean.append(np.mean(T_n))
# print(sample_mean)
Bars = plt.hist(sample_mean, label='n=5')
plt.show()
Y = []
X = []
for i in range(10):
    Y.append(Bars[0][i]/1000)
    X.append(Bars[2][i])
print(X)
# N_x = plt.hist(sample_mean, label='n=5')[1]
plt.bar(X, Y, width=1)
pdf = sts.norm.pdf(x, loc=mn, scale=SKO)
plt.plot(x, pdf, label='Нормальное распределение',  linewidth=3 )
plt.ylabel('Распределение выборочных средних')
plt.xlabel('$x_{cp}$')
plt.legend(loc='upper right')
plt.show()

# A = [10, 3, 6, 30, 49, 4, 4, 6, 90, 7]
# N = plt.hist(A, label='n=5') #, label='Выборка')
# plt.show()