import numpy as np
from sklearn import datasets
from sklearn import model_selection, linear_model, metrics
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic

blobs = datasets.make_blobs(300, centers=2, cluster_std=6, random_state=1)
plt.figure(figsize=(8,8))
plt.scatter(list(map(lambda x: x[0], blobs[0])), list(map(lambda x: x[1], blobs[0])),
            c=blobs[1], cmap='autumn')
# plt.show()

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(blobs[0], blobs[1],
                                                                                    test_size=15,
                                                                                    random_state=1)
ridge_model = linear_model.RidgeClassifier()
ridge_model.fit(train_data, train_labels)
print('roc_auc_score (ridge_model): ')
print(metrics.roc_auc_score(test_labels, ridge_model.predict(test_data)))

sgd_model = linear_model.SGDClassifier()
sgd_model.fit(train_data, train_labels)
print('roc_auc_score (sgd_model): ')
print(metrics.roc_auc_score(test_labels, sgd_model.predict(test_data)))

ridge_auc_scores = model_selection.cross_val_score(linear_model.RidgeClassifier(),
                                                 blobs[0], blobs[1], scoring='roc_auc', cv=20)
sgd_auc_scores = model_selection.cross_val_score(linear_model.SGDClassifier(max_iter=1000),
                                                 blobs[0], blobs[1], scoring='roc_auc', cv=20)

print('ridge model auc: mean {:.3f}, std {:.3f}'.format(ridge_auc_scores.mean(),
                                                        ridge_auc_scores.std(ddof=1)))
print('sgd model auc: mean {:.3f}, std {:.3f}'.format(sgd_auc_scores.mean(),
                                                        sgd_auc_scores.std(ddof=1)))

ridge_mean = ridge_auc_scores.mean()
sgd_mean = sgd_auc_scores.mean()

print('Со случайной (=0.25) дисперсией:')
print('ridge model mean auc 95% confidence interval: ', _zconfint_generic(ridge_mean,
                                                                           (0.25/len(ridge_auc_scores))**0.5,
                                                                           0.05,
                                                                           'two-sided'))
print('SGD model mean auc 95% confidence interval: ', _zconfint_generic(sgd_mean,
                                                                           (0.25/len(sgd_auc_scores))**0.5,
                                                                           0.05,
                                                                           'two-sided'))

ridge_mean_std = ridge_auc_scores.std(ddof=1)/(len(ridge_auc_scores))**0.5
sgd_mean_std = sgd_auc_scores.std(ddof=1)/(len(sgd_auc_scores))**0.5

print('С оцененной (по выборке) дисперсией:')

print('ridge model mean auc 95% confidence interval: ', _zconfint_generic(ridge_mean,
                                                                           ridge_mean_std,
                                                                           0.05,
                                                                           'two-sided'))
print('SGD model mean auc 95% confidence interval: ', _zconfint_generic(sgd_mean,
                                                                           sgd_mean_std,
                                                                           0.05,
                                                                           'two-sided'))