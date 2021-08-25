import pandas as pd
pd.__version__
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


circles = datasets.make_circles()
# print(circles)

print("features: {}".format(circles[0][:10]))
print("target: {}".format(circles[1][:10]))

colors = ListedColormap(['red', 'yellow'])
# colors = ListedColormap(['red', 'blue'])

plt.figure(figsize=(8,8))
plt.scatter(map(lambda x: x[0], circles[0]), map(lambda x: x[1], circles[0]), c=circles[1], cmap=colors)
