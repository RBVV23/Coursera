from matplotlib.colors import ListedColormap
from sklearn import model_selection, datasets, linear_model, metrics

import matplotlib.pyplot as plt
import numpy as np

data, target, coef =datasets.make_regression(n_features=2, n_informative=1, n_targets=1,
                                             noise=5, coef=True, random_state=2)
plt.scatter(list(map(lambda x: x[0], data)), target, color='r')
plt.scatter(list(map(lambda x: x[1], data)), target, color='b')
# plt.scatter(data[:,0], target, color = 'r')
# plt.scatter(data[:,1], target, color = 'b')
plt.show()

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, target,
                                                                                    test_size=0.3)
# Линейная регрессия

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(train_data, train_labels)
predictions = linear_regressor.predict(test_data)

print('test_labels = ', test_labels)
print('predictions =', predictions)

ans = metrics.mean_absolute_error(test_labels, predictions)
print(ans)

linear_scoring = model_selection.cross_val_score(linear_regressor, data, target,
                                                 scoring='neg_mean_absolute_error', cv=10)
print('mean: {}, std: {}'.format(linear_scoring.mean(), linear_scoring.std()))

scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=True)

linear_scoring = model_selection.cross_val_score(linear_regressor, data, target,
                                                 scoring=scorer, cv=10)
print('mean: {}, std: {}'.format(linear_scoring.mean(), linear_scoring.std()))

print('coef = ', coef)
print('linear_regressor.coef_', linear_regressor.coef_)
print('linear_regressor.intercept_', linear_regressor.intercept_)

print("y = {:.2f}*x1 + {:.2f}*x2".format(coef[0], coef[1]))
print("y = {:.2f}*x1 + {:.2f}*x2 + {:.2f}".format(linear_regressor.coef_[0],
                                                  linear_regressor.coef_[1],
                                                  linear_regressor.intercept_))

# Регрессия с регуляризацией L1 (Лассо)

lasso_regressor = linear_model.Lasso(random_state=3)
lasso_regressor.fit(train_data, train_labels)
lasso_predictions = lasso_regressor.predict(test_data)

lasso_scoring = model_selection.cross_val_score(lasso_regressor, data, target, scoring=scorer, cv=10)
print('mean: {}, std: {}'.format(lasso_scoring.mean(), lasso_scoring.std()))

print('lasso_regressor.coef = ', lasso_regressor.coef_)

print("y = {:.2f}*x1 + {:.2f}*x2".format(coef[0], coef[1]))
print("y = {:.2f}*x1 + {:.2f}*x2".format(lasso_regressor.coef_[0], lasso_regressor.coef_[1]))
