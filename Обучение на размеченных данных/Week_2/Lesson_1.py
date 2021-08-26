from pandas import DataFrame
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

circles = datasets.make_circles()
# print(circles)

print("features: {}".format(circles[0][:10]))
print("target: {}".format(circles[1][:10]))

colors = ListedColormap(['red', 'yellow'])
# colors = ListedColormap(['red', 'blue'])


plot_2d_dataset(circles, colors)

noisy_circles = datasets.make_circles(noise=0.15)
plot_2d_dataset(noisy_circles, colors)


colors = ListedColormap(['red', 'blue', 'green', 'yellow'])

simple_classification_problem = datasets.make_classification(n_features=2, n_informative=1, n_classes=2,
                                                             n_redundant=1, n_clusters_per_class=1,
                                                             random_state=1)
plot_2d_dataset(simple_classification_problem, colors)

simple_classification_problem = datasets.make_classification(n_features=2, n_informative=2, n_classes=4,
                                                             n_redundant=0, n_clusters_per_class=1,
                                                             random_state=1)
plot_2d_dataset(simple_classification_problem, colors)


iris = datasets.load_iris()
print(iris)
print(iris.keys())
print(iris.DESCR)

print('feature names: {}'.format(iris.feature_names))
print('target names: {names}'.format(names=iris.target_names))

print(iris.data[:10])
print(iris.target)

iris_frame = DataFrame(iris.data)
iris_frame.columns = iris.feature_names
iris_frame['target'] = iris.target
# iris_frame['target2'] = iris.target

print(iris_frame.head())

iris_frame['target'] = iris_frame['target'].apply(lambda x: iris.target_names[x])
# iris_frame['target2'] = iris_frame['target2'].apply(lambda x: iris.target_names[x % 2])
print(iris_frame.head())

iris_frame[iris_frame.target == 'setosa'].hist('sepal length (cm)')
# plt.show()

plt.figure(figsize=(20, 24))
plot_number = 0
for feature_name in iris.feature_names:
    for target_name in iris['target_names']:
        plot_number += 1
        plt.subplot(4, 3, plot_number)
        plt.hist(iris_frame[iris_frame.target == target_name][feature_name])
        plt.title(target_name)
        plt.xlabel('cm')
        plt.ylabel(feature_name[:-4])
# plt.show()

# sns.set(font_scale=1.3)
sns.pairplot(iris_frame, hue='target')
# plt.show()

data = sns.load_dataset('iris')
sns.pairplot(data, hue='species')
plt.show()