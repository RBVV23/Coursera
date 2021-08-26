import pandas as pd
pd.__version__
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_2d_dataset(data, colors):
    plt.figure(figsize=(8,8))
    plt.scatter(list(map(lambda x: x[0], data[0])), list(map(lambda x: x[1], data[0])), c=data[1], cmap=colors)
    plt.show()

circles = datasets.make_circles()
## print(circles)

# print("features: {}".format(circles[0][:10]))
# print("target: {}".format(circles[1][:10]))

colors = ListedColormap(['red', 'yellow'])
# colors = ListedColormap(['red', 'blue'])


# plot_2d_dataset(circles, colors)

noisy_circles = datasets.make_circles(noise=0.15)
# plot_2d_dataset(noisy_circles, colors)


colors = ListedColormap(['red', 'blue', 'green', 'yellow'])

simple_classification_problem = datasets.make_classification(n_features=2, n_informative=1, n_classes=2,
                                                             n_redundant=1, n_clusters_per_class=1,
                                                             random_state=1)
# plot_2d_dataset(simple_classification_problem, colors)

simple_classification_problem = datasets.make_classification(n_features=2, n_informative=2, n_classes=4,
                                                             n_redundant=0, n_clusters_per_class=1,
                                                             random_state=1)
# plot_2d_dataset(simple_classification_problem, colors)


iris = datasets.load_iris()
# print(iris)
print(iris.keys())
print(iris.DESCR)

print('fe')