from matplotlib.colors import ListedColormap
from sklearn import model_selection, datasets, linear_model, metrics

import matplotlib.pyplot as plt
import numpy as np

blobs = datasets.make_blobs(centers=2, cluster_std=5.5, random_state=1)
colors = ListedColormap(['red', 'blue'])

plt.figure(figsize=(6,6))
plt.scatter([x[0] for x in blobs[0]], [x[1] for x in blobs[0]], c=blobs[1], cmap=colors)
# plt.scatter(list(map(lambda x: x[0], blobs[0])), list(map(lambda x: x[1], blobs[0])), c=blobs[1], cmap=colors)
plt.show()

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(blobs[0], blobs[1],
                                                                                  test_size=0.3, random_state=1)

# Линейная классификация

ridge_classifer = linear_model.RidgeClassifier(random_state=1)
ridge_classifer.fit(train_data, train_labels)

ridge_predictions = ridge_classifer.predict(test_data)

print(test_labels)
print(ridge_predictions)

ans = metrics.accuracy_score(test_labels, ridge_predictions)
print(ans)

print('ridge_classifer.coef_ = ', ridge_classifer.coef_)
print('ridge_classifer.intercept_ = ', ridge_classifer.intercept_)

# Логистическая регрессия

log_regressor = linear_model.LogisticRegression(random_state=1)
log_regressor.fit(train_data, train_labels)
log_predictions = log_regressor.predict(test_data)
log_proba_predictions = log_regressor.predict_proba(test_data)

print('\ttest_labels: ', test_labels)
print('\tlog_predictions: ', log_predictions)
print('\tlog_proba_predictions: ', log_proba_predictions)

ans = metrics.accuracy_score(test_labels, log_predictions)
print(ans)

# Кросс-валидация

ridge_scoring = model_selection.cross_val_score(ridge_classifer, blobs[0], blobs[1],
                                                scoring='accuracy', cv=10)
log_scoring = model_selection.cross_val_score(log_regressor, blobs[0], blobs[1],
                                                scoring='accuracy', cv=10)

print('ridge_scoring = ', ridge_scoring)
print('Ridge mean: {}, max: {}, min: {}, std: {}'.format(ridge_scoring.mean(), ridge_scoring.max(),
                                                         ridge_scoring.min(), ridge_scoring.std()))

print('log_scoring = ', log_scoring)
print('Log mean: {}, max: {}, min: {}, std: {}'.format(log_scoring.mean(), log_scoring.max(),
                                                         log_scoring.min(), log_scoring.std()))

scorer = metrics.make_scorer(metrics.accuracy_score)
cv_strategy = model_selection.StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=2)
cv_strategy.get_n_splits(blobs[1])

ridge_scoring = model_selection.cross_val_score(ridge_classifer, blobs[0], blobs[1],
                                                scoring=scorer, cv=cv_strategy)
log_scoring = model_selection.cross_val_score(log_regressor, blobs[0], blobs[1],
                                                scoring=scorer, cv=cv_strategy)

# print('ridge_scoring = ', ridge_scoring)
print('Ridge mean: {}, max: {}, min: {}, std: {}'.format(ridge_scoring.mean(), ridge_scoring.max(),
                                                         ridge_scoring.min(), ridge_scoring.std()))

# print('log_scoring = ', log_scoring)
print('Log mean: {}, max: {}, min: {}, std: {}'.format(log_scoring.mean(), log_scoring.max(),
                                                         log_scoring.min(), log_scoring.std()))
