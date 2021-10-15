import numpy as np
from scipy import stats
import scipy
import matplotlib.pyplot as plt

n = 16
F_H0 = stats.binom(n, 0.5)

x = np.linspace(0, 16, 17)

plt.bar(x, F_H0.pmf(x), align='center')
plt.xlim(-0.5, 16.5)
plt.title('Гистограмма распределения Бернулли')
plt.show()

# print('x = ', x)
# print('F_H0 = ', F_H0)
# print('F_H0.pmf(x) = ', F_H0.pmf(x))

plt.bar(x, F_H0.pmf(x), align='center')
plt.bar(np.linspace(12,16,5), F_H0.pmf(np.linspace(12,16,5)), align='center', color='red')
plt.xlim(-0.5, 16.5)
plt.show()

print('Вероятность получить больше 12 положительных результатов при условии, что нулевая гипотеза (p = 0.5) верна:')
print(stats.binom_test(12,16,0.5, alternative='greater')*100, '%')

plt.bar(x, F_H0.pmf(x), align='center')
plt.bar(np.linspace(11,16,6), F_H0.pmf(np.linspace(11,16,6)), align='center', color='red')
plt.xlim(-0.5, 16.5)
plt.show()

print('Вероятность получить больше 11 положительных результатов при условии, что нулевая гипотеза (p = 0.5) верна:')
print(stats.binom_test(11,16,0.5, alternative='greater')*100, '%')

plt.bar(x, F_H0.pmf(x), align='center')
plt.bar(np.linspace(12,16,5), F_H0.pmf(np.linspace(12,16,5)), align='center', color='red')
plt.bar(np.linspace(0,4,5), F_H0.pmf(np.linspace(0,4,5)), align='center', color='red')
plt.show()

print('Вероятность получить больше 12 одинаковых результатов при условии, что нулевая гипотеза (p = 0.5) верна:')
print(stats.binom_test(12,16,0.5, alternative='two-sided')*100, '%')


plt.bar(x, F_H0.pmf(x), align='center')
plt.bar(np.linspace(13,16,4), F_H0.pmf(np.linspace(13,16,4)), align='center', color='red')
plt.bar(np.linspace(0,3,4), F_H0.pmf(np.linspace(0,3,4)), align='center', color='red')
plt.show()

print('Вероятность получить больше 13 одинаковых результатов при условии, что нулевая гипотеза (p = 0.5) верна:')
print(stats.binom_test(13,16,0.5, alternative='two-sided')*100, '%')