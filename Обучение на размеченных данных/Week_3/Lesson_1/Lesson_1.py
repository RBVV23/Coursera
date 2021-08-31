from sklearn import model_selection, datasets, linear_model, metrics
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

iris = datasets.load_iris()
# print(iris)
print(iris.data)
print(iris.target)

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(iris.data,
                                                                                    iris.target,
                                                                                    test_size=0.3,
                                                                                    random_state=0)

classifer = linear_model.SGDClassifier(random_state=0)

print('ПАРАМЕТРЫ ПО УМОЛЧАНИЮ:')
for param in classifer.get_params().keys():
    print('{} = {}'.format(param, classifer.get_params()[param]))
print()

parametrs_grid = {'loss': ['hinge', 'log', 'squared_hinge', 'squared_loss'],
                  'penalty': ['l1', 'l2'],
                  'max_iter': np.arange(5, 10),
                  'alpha': np.linspace(0.0001, 0.001, num=5)}
# print(type(parametrs_grid))

cv = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# print(type(cv))

grid_cv = model_selection.GridSearchCV(classifer, parametrs_grid, scoring='accuracy', cv=cv)
grid_cv.fit(train_data, train_labels)

print('grid_cv.best_estimator_: ', grid_cv.best_estimator_)
print('grid_cv.best_score_: ', grid_cv.best_score_)
print('grid_cv.best_params_: ', grid_cv.best_params_)

# print('grid_cv.cv_results_:\n', grid_cv.cv_results_)

randomize_grid_cv = model_selection.RandomizedSearchCV(classifer, parametrs_grid,
                                                       scoring='accuracy', cv=cv,
                                                       n_iter=20, random_state=0)
randomize_grid_cv.fit(train_data, train_labels)
print()
print('grid_cv.best_estimator_: ', grid_cv.best_estimator_)
print('grid_cv.best_score_: ', grid_cv.best_score_)
print('grid_cv.best_params_: ', grid_cv.best_params_)