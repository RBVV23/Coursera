import numpy as np
from pandas import DataFrame
from sklearn import model_selection, datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_2d_dataset(data, colors):
    plt.figure(figsize=(8,8))
    plt.scatter(list(map(lambda x: x[0], data[0])), list(map(lambda x: x[1], data[0])), c=data[1], cmap=colors)
    plt.show()

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

iris = datasets.load_iris()

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(iris.data,
                                                                iris.target, test_size=0.3)
print(len(test_labels)/len(iris.data))
print('Размер обучающей выборки: {} объектов'.format(len(train_data)))
print('Размер тестовой выборки: {} объектов'.format(len(test_data)))

print('Обучающая выборка: \n', train_data[:5], '\n')
print('Тестовая выборка: \n', test_data[:5], '\n')

print('Метки классов на обучающей выборке: \n', train_labels, '\n')
print('Метки классов на тестовой выборке: \n', test_labels, '\n')


X = range(10)
kf = model_selection.KFold(n_splits=5)
print('\nKFold:')
for train_indices, test_indices in kf.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

kf = model_selection.KFold(n_splits=2, shuffle=True)
print('\nKFold (shuffle=True):')
for train_indices, test_indices in kf.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

kf = model_selection.KFold(n_splits=2, shuffle=True, random_state=1)
print('\nKFold (random_state=1):')
for train_indices, test_indices in kf.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

target = np.array([0]*5 + [1]*5)
print('\n', target)

skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
print('\nStratifiedKFold:')
for train_indices, test_indices in skf.split(X, target):

    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

ss = model_selection.ShuffleSplit(n_splits = 10, test_size=0.2)
print('\nShuffleSplit:')
for train_indices, test_indices in ss.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

sss = model_selection.StratifiedShuffleSplit(n_splits = 4, test_size=0.2)
print('\nStratifiedShuffleSplit:')
for train_indices, test_indices in sss.split(X, target):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

loo = model_selection.LeaveOneOut()
print('\nLeaveOneOut:')
for train_indices, test_indices in loo.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))
