import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt
print(plt.__version__)
import scipy.stats as sts
import pandas as pd
print(pd.__version__)
from collections import Counter

T = np.random.choice([1, 2, 3, 4, 5, 6], 100)
print(T)
c = Counter(T)
print("Число выпадений каждой из сторон:")
print(c)
print("Вероятности выпадений каждой из сторон:")
print(c.items())
print()
for k, v in c.items():
    print('k=' + str(k) + ' v=' + str(v/100.0))
print({k: v/100 for k, v in c.items()})

T = sts.norm(1,10).rvs(100)
# T_2 = T.rvs(100)
print(T)

x = np.linspace(-4,4,100)
pdf = sts.norm(0,1).pdf(x)
plt.plot(x, pdf, label='Theoretical PDF')

plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.legend(loc='upper left')
plt.show()

plt.hist(T, density=True)
plt.ylabel('Fraction of samples')
plt.xlabel('$x$')
plt.show()

plt.hist(T, bins=3, density=True)
plt.ylabel('Fraction of samples')
plt.xlabel('$x$')
plt.show()



