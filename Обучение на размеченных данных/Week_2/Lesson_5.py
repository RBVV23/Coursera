from sklearn import metrics, datasets, model_selection, linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

clf_data, clf_target = datasets.make_classification(n_features=2, n_informative=2, n_classes=2,
                                                    n_redundant=0, n_clusters_per_class=1,
                                                    random_state=7)
reg_data, reg_target = datasets.make_regression(n_features=2, n_informative=1, n_targets=1,
                                                    noise=5., random_state=7)

colors = ListedColormap(['red', 'blue'])
# plt.scatter(clf_data[:,0], clf_data[:,1], c=clf_target, cmap=colors)
# plt.show()
# plt.scatter(reg_data[:,0], reg_target, color='r')
# plt.scatter(reg_data[:,1], reg_target, color='b')
# plt.show()

# print(clf_data.shape)
# print(clf_target.shape)

clf_train_data, clf_test_data, clf_train_labels, clf_test_labels = model_selection.train_test_split(clf_data,
                                                                                                    clf_target,
                                                                                                    test_size=0.3,
                                                                                                    random_state=1)
reg_train_data, reg_test_data, reg_train_labels, reg_test_labels = model_selection.train_test_split(reg_data,
                                                                                                    reg_target,
                                                                                                    test_size=0.3,
                                                                                                    random_state=1)
classifer = linear_model.SGDClassifier(loss='log', random_state=1, max_iter=1000)
classifer.fit(clf_train_data, clf_train_labels)
predictions = classifer.predict(clf_test_data)
proba_predictions = classifer.predict_proba(clf_test_data)

my_proba = []
n = 0
for i in proba_predictions:
    my_proba.append(i[predictions[n]])
    n += 1

print('clf_test_labels = ', clf_test_labels)
print('predictions = ', predictions)
print('proba_predictions = ', proba_predictions)
print('my_proba = ', my_proba)

my_accuracy = sum([1. if pair[0] == pair[1] else 0. for pair in zip(clf_test_labels, predictions)])/len(clf_test_labels)
print('my_accuracy = ', my_accuracy)
accuracy = metrics.accuracy_score(clf_test_labels, predictions)
print('accuracy = ', accuracy)

matrix = metrics.confusion_matrix(clf_test_labels, predictions)
print(matrix)
d = sum([1. if pair[0] == pair[1] else 0. for pair in zip(clf_test_labels, predictions)])
print(d - matrix.diagonal().sum())

pr_0 = metrics.precision_score(clf_test_labels, predictions, pos_label=0)
pr_1 = metrics.precision_score(clf_test_labels, predictions, pos_label=1)
print(pr_0, pr_1)

rc_0 = metrics.recall_score(clf_test_labels, predictions, pos_label=0)
rc_1 = metrics.recall_score(clf_test_labels, predictions, pos_label=1)
print(rc_0, rc_1)

f_0 = metrics.f1_score(clf_test_labels, predictions, pos_label=0)
f_1 = metrics.f1_score(clf_test_labels, predictions, pos_label=1)
print(f_0, f_1)

print(metrics.classification_report(clf_test_labels, predictions))

fpr, tpr, _ = metrics.roc_curve(clf_test_labels, proba_predictions[:,1])
# print(_)
plt.plot(fpr, tpr, label='linear model')
plt.plot([0,1], [0,1], '--', color='grey', label='random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
# plt.show()

S1 = metrics.roc_auc_score(clf_test_labels, predictions)
S2 = metrics.roc_auc_score(clf_test_labels, proba_predictions[:,1])
print(S1, S2, str(abs(S2-S1)))

AUC = metrics.average_precision_score(clf_test_labels, predictions)
print('AUC = ', AUC)
log_loss = metrics.log_loss(clf_test_labels, proba_predictions[:,1])
print('log_loss = ', log_loss)

regressor = linear_model.SGDRegressor(random_state=1, max_iter=20)
regressor.fit(reg_train_data, reg_train_labels)
reg_predictions = regressor.predict(reg_test_data)

print(reg_test_labels)
print(reg_predictions)

MAE = metrics.mean_absolute_error(reg_test_labels, reg_predictions)
print('MAE = ', MAE)
MSE = metrics.mean_squared_error(reg_test_labels, reg_predictions)
print('MSE = ', MSE)
print('root(MSE) = ', str(MSE**0.5))
R2 = metrics.r2_score(reg_test_labels, reg_predictions)
print('R2 = ', R2)