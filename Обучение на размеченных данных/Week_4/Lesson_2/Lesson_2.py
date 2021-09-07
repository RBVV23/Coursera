from sklearn import ensemble, model_selection, metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

bioresponse = pd.read_csv('bioresponse.csv', header=0, sep=',')

print(bioresponse.head())
print('bioresponse.shape = ', bioresponse.shape)
print('bioresponse.columns:')
print(bioresponse.columns)

bioresponse_target = bioresponse.Activity.values
sum_ones = sum(bioresponse_target)/float(len(bioresponse_target))
print('bioresponse = 1: {:.2f}\nbioresponse = 0: {:.2f}'.format(sum_ones, 1 - sum_ones)

bioresponse_data = bioresponse.iloc[:,1:]

rf_classifer_low_depth = ensemble.RandomForestClassifier(n_estimators=50, max_depth=2, random_state=1)
train_sizes, train_scores, test_scores = model_selection.learning_curve(rf_classifer_low_depth,
                                                                        bioresponse_data, bioresponse_target,
                                                                        train_sizes=np.arange(0.1, 1.0, 0.2),
                                                                        cv=3, scoring='accuracy')
print('train_sizes: ', train_sizes)
print('train_scores: ', train_scores)
