from matplotlib.colors import ListedColormap
from sklearn import model_selection, datasets, metrics, tree
import numpy as np
import matplotlib.pyplot as plt

def get_meshgrid(data, step=0.05, border=0.5):
    x_min = data[:, 0].min() - border
    x_max = data[:, 0].max() + border
    y_min = data[:, 1].min() - border
    y_max = data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

def plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels,
                          colors=ListedColormap(['red', 'blue', 'yellow']),
                          light_colors=ListedColormap(['lightcoral', 'lightblue', 'lightyellow'])):

    estimator.fit(train_data, train_labels)

    plt.figure(figsize=(16,6))

    plt.subplot(1,2,1)
    xx, yy = get_meshgrid(train_data)
    mesh_predictions = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors, shading='auto')
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=colors, s=100)
    accuracy = metrics.accuracy_score(train_labels, estimator.predict(train_data))
    plt.title('Train data accuracy = {}'.format(accuracy))

    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors, shading='auto')
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=colors, s=100)
    accuracy = metrics.accuracy_score(test_labels, estimator.predict(test_data))
    plt.title('Test data accuracy = {}'.format(accuracy))

    plt.show()


classification_problem = datasets.make_classification(n_features=2, n_informative=2, n_redundant=0,
                                                      n_classes=3, n_clusters_per_class=1, random_state=3)

colors = ListedColormap(['red', 'blue', 'yellow'])
light_colors = ListedColormap(['lightcoral', 'lightblue', 'lightyellow'])

plt.figure(figsize=(8,6))
plt.scatter(list(map(lambda x: x[0], classification_problem[0])),
            list(map(lambda x: x[1], classification_problem[0])), c=classification_problem[1],
                     cmap=colors, s=100)
plt.show()
print('classification_problem: ')
print(classification_problem)

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(classification_problem[0],
                                                                                    classification_problem[1],
                                                                                    test_size=0.3,
                                                                                    random_state=1)

clf = tree.DecisionTreeClassifier(random_state=1)
clf.fit(train_data, train_labels)
predictions = clf.predict(test_data)
accuracy = metrics.accuracy_score(test_labels, predictions)
print('accuracy = ', accuracy)
print('predictions: ', predictions)

estimator = tree.DecisionTreeClassifier(random_state=1, max_depth=1)
plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels)

estimator = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels)

estimator = tree.DecisionTreeClassifier(random_state=1, max_depth=3)
plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels)

estimator = tree.DecisionTreeClassifier(random_state=1)
plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels)

estimator = tree.DecisionTreeClassifier(random_state=1, min_samples_leaf=3)
plot_decision_surface(estimator, train_data, train_labels, test_data, test_labels)