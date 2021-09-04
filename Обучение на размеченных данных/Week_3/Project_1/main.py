import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

def calculate_means(numeric_data):
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull().values)[0] # - авторская версия
        correction = np.amax(to_sum[indices])
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
    return pd.Series(means, numeric_data.columns)


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 100)


data = pd.read_csv('data.csv')
print(data.shape)

print(data.head(100))



# place your code here

cv = 3

X_train= [0,0,0,0,0,0,0,0,0]
y_train= [0,0,0,0,0,0,0,0,0]
X_test= [0,0,0,0,0,0,0,0,0]
y_test= [0,0,0,0,0,0,0,0,0]


param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

estimator = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
optimizer.fit(X_train, y_train)

auc = roc_auc_score(y_test, optimizer.predict_proba(X_test)[:,1])
print('auc = ', auc)
# print('weights = ', optimizer._coef)
estimator.coef_
# write_answer_6(auc)


