import  numpy as np
from sklearn import datasets
from sklearn import model_selection, linear_model, metrics
import matplotlib.pyplot as plt

blobs = datasets.make_blobs(300, centers=2, cluster_std=6, random_state=1)
plt.figure(figsize=(8,8))
plt.scatter(list(map(lambda x: x[0], blobs[0])), list(map(lambda x: x[1], blobs[0])),
            c=blobs[1], cmap='autumn')
plt.show()

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
