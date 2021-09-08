from sklearn import ensemble, model_selection, metrics
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
print(xgb.__version__)

bioresponce = pd.read_csv('bioresponse.csv', header=0, sep=',')

print(bioresponce.head())

bioresponce_target = bioresponce.Activity.values
bioresponce_data = bioresponce.iloc[:, 1:]

n_trees = [1] + list(range(10,55,5))
# print('n_trees: ', n_trees)
scoring = []
for n_tree in n_trees:
    estimator = ensemble.RandomForestClassifier(n_estimators=n_tree, min_samples_split=5, random_state=1)
    score = model_selection.cross_val_score(estimator, bioresponce_data, bioresponce_target,
                                            scoring='accuracy', cv=3)
    scoring.append(score)
scoring = np.asmatrix(scoring)
print('scoring:')
print(scoring)

# print(scoring.mean(axis=1).shape)


xgb_scoring = []
for n_tree in n_trees:
    # estimator = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=n_tree, min_child_weight=3)
    estimator = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=n_tree, min_child_weight=3,
                                  use_label_encoder=False)  # use_label_encoder=False - позволит избежать
                                                            # ошибок в будущих версиях библиотеки xgboost
    score = model_selection.cross_val_score(estimator, bioresponce_data, bioresponce_target,
                                            scoring='accuracy', cv=3)
    xgb_scoring.append(score)
xgb_scoring = np.asmatrix(xgb_scoring)
print('xgb_scoring:')
print(xgb_scoring)

plt.plot(n_trees, scoring.mean(axis=1), marker='.', label='RandomForest')
plt.plot(n_trees, xgb_scoring.mean(axis=1), marker='.', label='XGBoost')
plt.grid(True)
plt.xlabel('n_trees')
plt.ylabel('score')
plt.title('Accuracy score')
plt.legend(loc='lower right')
plt.show()