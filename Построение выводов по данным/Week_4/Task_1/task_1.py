from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m

plt.figure(figsize=(8,8))

Xt = np.arange(-10, 11, 1)
Yt = np.arange(-10, 11, 1)
x = np.linspace(-9, 9, 1000)
# print(x)
# y = -1 + np.sqrt(x+2)
# y = -4*np.exp(-x/3)
# y = 1 - np.log(x-4)/np.log(3)
# y = 0.5*np.sin(x + np.pi/4)

k = -0.2
y = np.exp(k*x)

Xmax = max(Xt)
Xmin = min(Xt)
Ymax = max(Yt)
Ymin = min(Yt)


# plt.ax.arrow(0, 0, 1, 1)

plt.plot(x, y, color='r')
plt.plot([Xmin,Xmax],[0,0], color='k')
plt.plot([0,0],[Ymin,Ymax], color='k')
plt.xticks(Xt)
plt.yticks(Yt)
plt.grid()
plt.xlim(Xmin, Xmax)
plt.ylim(Ymin, Ymax)

plt.show()
