import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


import warnings
warnings.filterwarnings('ignore')

def calculate_means(numeric_data):
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
#       indices = np.nonzero(~numeric_data.iloc[:,j].isnull())[0] - оригинальная строка (не работает)
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull().values)[0] # - авторская версия
        correction = np.amax(to_sum[indices])
#         to_sum /= correction - первая строка из ненужной пары
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
#         means[j] *= correction - вторая строка из ненужной пары
    return pd.Series(means, numeric_data.columns)
def plot_scores(optimizer):
    scores = []
    for i in range(len(optimizer.cv_results_['params'])):
        scores.append([optimizer.cv_results_['params'][i]['C'],
                       optimizer.cv_results_['mean_test_score'][i],
                       optimizer.cv_results_['std_test_score'][i]])
    scores = np.array(scores)
    plt.semilogx(scores[:, 0], scores[:, 1])
    plt.fill_between(scores[:, 0], scores[:, 1] - scores[:, 2],
                     scores[:, 1] + scores[:, 2], alpha=0.3)
    plt.show()

def write_answer_1(auc_1, auc_2):
    auc = (auc_1 + auc_2) / 2
    with open("preprocessing_lr_answer1.txt", "w") as fout:
        fout.write(str(auc))
def write_answer_2(auc):
    with open("preprocessing_lr_answer2.txt", "w") as fout:
        fout.write(str(auc))
def write_answer_3(auc_1, auc_2):
    auc = (auc_1 + auc_2) / 2
    with open("preprocessing_lr_answer3.txt", "w") as fout:
        fout.write(str(auc))
def write_answer_4(auc):
    with open("preprocessing_lr_answer4.txt", "w") as fout:
        fout.write(str(auc))
def write_answer_5(auc):
    with open("preprocessing_lr_answer5.txt", "w") as fout:
        fout.write(str(auc))
def write_answer_6(features):
    with open("preprocessing_lr_answer6.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in features]))

data = pd.read_csv('data.csv')
print('data.shape = ', data.shape)

X = data.drop('Grant.Status', 1)
y = data['Grant.Status']

print('data.head()')
data.head()

numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3',
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))

data.dropna().shape

means = data[numeric_cols].mean()
print('means = ', means)

numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3',
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']

means = data[numeric_cols].mean()

X = data.drop('Grant.Status', 1)
# y = data['Grant.Status']
X_real_mean = X[numeric_cols]
# X_real_zeros = X

for col in numeric_cols:
    indices = X[col].isnull()
    X_real_mean[col][indices] = X_real_mean[col][indices].apply(lambda x: means[col])


X = data.drop('Grant.Status', 1)
X_real_zeros = X[numeric_cols]
for col in numeric_cols:
    indices = X[col].isnull()
    X_real_zeros[col][indices] = X_real_zeros[col][indices].apply(lambda x: 0)

X = data.drop('Grant.Status', 1)
X_cat = X[categorical_cols]
for col in categorical_cols:
    indices = X[col].isnull()
    X_cat[col][indices] = X_cat[col][indices].apply(lambda x: 'NA')
    X_cat[col] = X_cat[col].apply(lambda x: str(x))

X = data.drop('Grant.Status', 1)
print('X_cat', X_cat)

categorial_data = pd.DataFrame({'sex': ['male', 'female', 'male', 'female'],
                                'nationality': ['American', 'European', 'Asian', 'European']})
print('Исходные данные:\n')
print(categorial_data)
encoder = DV(sparse = False)
encoded_data = encoder.fit_transform(categorial_data.T.to_dict().values())
print('\nЗакодированные данные:\n')
print(encoded_data)

encoder = DV(sparse = False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())

(X_train_real_zeros,
 X_test_real_zeros,
 y_train, y_test) = train_test_split(X_real_zeros, y,
                                     test_size=0.3,
                                     random_state=0)
(X_train_real_mean,
 X_test_real_mean) = train_test_split(X_real_mean,
                                      test_size=0.3,
                                      random_state=0)
(X_train_cat_oh,
 X_test_cat_oh) = train_test_split(X_cat_oh,
                                   test_size=0.3,
                                   random_state=0)

X_train_m = np.hstack((X_train_real_mean, X_train_cat_oh))
X_test_m = np.hstack((X_test_real_mean, X_test_cat_oh))

X_train_z = np.hstack((X_train_real_zeros, X_train_cat_oh))
X_test_z = np.hstack((X_test_real_zeros, X_test_cat_oh))

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

estimator = LogisticRegression(penalty='l2', solver='liblinear')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')



optimizer.fit(X_train_m, y_train)

print(optimizer.best_estimator_)
print('X_real_mean: ', optimizer.best_score_)
plt.title('X_real_mean')
plot_scores(optimizer)
auc_1 = roc_auc_score(y_test, optimizer.predict_proba(X_test_m)[:,1])
print('auc_1 = ', auc_1)


optimizer.fit(X_train_z, y_train)

print(optimizer.best_estimator_)
print('X_real_zeros: ', optimizer.best_score_)
plt.title('X_real_zeros')
plot_scores(optimizer)
plt.show()
auc_2 = roc_auc_score(y_test, optimizer.predict_proba(X_test_z)[:,1])
print('auc_2 = ', auc_2)

write_answer_1(auc_1, auc_2)
print('(auc_1 + auc_2)/2 = ', (auc_1 + auc_2)/2)


data_numeric = pd.DataFrame(X_train_real_zeros, columns=numeric_cols)
list_cols = ['Number.of.Successful.Grant.1', 'SEO.Percentage.2', 'Year.of.Birth.1']
scatter_matrix(data_numeric[list_cols], alpha=0.5, figsize=(10, 10))
plt.show()

scaler = StandardScaler()

X_train_real_scaled = scaler.fit_transform(X_train_real_zeros, y_train)
X_test_real_scaled = scaler.transform(X_test_real_zeros)

X_train_real_scaled_m = scaler.fit_transform(X_train_real_mean, y_train)
X_test_real_scaled_m = scaler.transform(X_test_real_mean)

data_numeric_scaled = pd.DataFrame(X_train_real_scaled, columns=numeric_cols)
list_cols = ['Number.of.Successful.Grant.1', 'SEO.Percentage.2', 'Year.of.Birth.1']
scatter_matrix(data_numeric_scaled[list_cols], alpha=0.5, figsize=(10, 10))
plt.show()


param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

X_train_z = np.hstack((X_train_real_scaled, X_train_cat_oh))
X_test_z = np.hstack((X_test_real_scaled, X_test_cat_oh))

estimator = LogisticRegression(penalty='l2', solver='liblinear')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')

optimizer.fit(X_train_z, y_train)

print(optimizer.best_estimator_)
print('X_real_z_scaled: ', optimizer.best_score_)
plt.title('X_real_z_scaled')
plot_scores(optimizer)
auc_z = roc_auc_score(y_test, optimizer.predict_proba(X_test_z)[:, 1])
print('auc_z = ', auc_z)

X_train_m = np.hstack((X_train_real_scaled_m, X_train_cat_oh))
X_test_m = np.hstack((X_test_real_scaled_m, X_test_cat_oh))

estimator = LogisticRegression(penalty='l2', solver='liblinear')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')

optimizer.fit(X_train_m, y_train)

print(optimizer.best_estimator_)
print('X_real_m_scaled: ', optimizer.best_score_)
plt.title('X_real_m_scaled')
plot_scores(optimizer)
plt.show()
auc_m = roc_auc_score(y_test, optimizer.predict_proba(X_test_m)[:, 1])
print('auc_m = ', auc_m)

auc = max(auc_m, auc_z)
write_answer_2(auc)


print('np.sum(y_train==0) = ', np.sum(y_train==0))
print('np.sum(y_train==1) = ', np.sum(y_train==1))

np.random.seed(0)
"""Сэмплируем данные из первой гауссианы"""
data_0 = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], size=40)
"""И из второй"""
data_1 = np.random.multivariate_normal([0,1], [[0.5,0],[0,0.5]], size=40)
"""На обучение берём 20 объектов из первого класса и 10 из второго"""
example_data_train = np.vstack([data_0[:20,:], data_1[:10,:]])
example_labels_train = np.concatenate([np.zeros((20)), np.ones((10))])
"""На тест - 20 из первого и 30 из второго"""
example_data_test = np.vstack([data_0[20:,:], data_1[10:,:]])
example_labels_test = np.concatenate([np.zeros((20)), np.ones((30))])
"""Задаём координатную сетку, на которой будем вычислять область классификации"""
xx, yy = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
"""Обучаем регрессию без балансировки по классам"""
optimizer = GridSearchCV(LogisticRegression(), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train, example_labels_train)
"""Строим предсказания регрессии для сетки"""
Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
"""Считаем AUC"""
auc_wo_class_weights = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
plt.title('Without class weights')
plt.show()
print('AUC: %f'%auc_wo_class_weights)
"""Для второй регрессии в LogisticRegression передаём параметр class_weight='balanced'"""
optimizer = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train, example_labels_train)
Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
auc_w_class_weights = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
plt.title('With class weights')
plt.show()
print('AUC: %f'%auc_w_class_weights)

X_train = X_train_m
X_test = X_test_m
y_train_long = np.array(y_train)

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

estimator = LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
optimizer.fit(X_train, y_train)

print(optimizer.best_estimator_)
print('X_real_m_scaled (auto_balance): ', optimizer.best_score_)
plt.title('X_real_m_scaled (auto_balance)')
plot_scores(optimizer)
auc_ab = roc_auc_score(y_test, optimizer.predict_proba(X_test)[:, 1])
print('auc_ab = ', auc_ab)
auc_1 = auc_ab

dN = abs(np.sum(y_train == 1) - np.sum(y_train == 0)) // 2

np.random.seed(0)

for i in range(dN):
    indices_to_add = np.random.randint(np.sum(y_train == 1))
    X_train_to_add = X_train[np.array(y_train) == 1, :][indices_to_add, :]
    y_train_to_add = y_train_long[indices_to_add]
    np.vstack((X_train, X_train_to_add))
    np.hstack((y_train_long, y_train_to_add))

estimator = LogisticRegression(penalty='l2', solver='liblinear')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
optimizer.fit(X_train, y_train_long)

print(optimizer.best_estimator_)
print('X_real_m_scaled (manually_balance): ', optimizer.best_score_)
plt.title('X_real_m_scaled (manually_balance)')
plot_scores(optimizer)
auc_mb = roc_auc_score(y_test, optimizer.predict_proba(X_test)[:, 1])
print('auc_mb = ', auc_mb)
auc_2 = auc_mb


write_answer_3(auc_1, auc_2)

print('AUC ROC for classifier without weighted classes', auc_wo_class_weights)
print('AUC ROC for classifier with weighted classes: ', auc_w_class_weights)

"""Разделим данные по классам поровну между обучающей и тестовой выборками"""
example_data_train = np.vstack([data_0[:20,:], data_1[:20,:]])
example_labels_train = np.concatenate([np.zeros((20)), np.ones((20))])
example_data_test = np.vstack([data_0[20:,:], data_1[20:,:]])
example_labels_test = np.concatenate([np.zeros((20)), np.ones((20))])
"""Обучим классификатор"""
optimizer = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train, example_labels_train)
Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
auc_stratified = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
plt.title('With class weights')
plt.show()
print('AUC ROC for stratified samples: ', auc_stratified)

X_train_z, X_test_z, y_train, y_test = train_test_split(X_real_zeros, y, stratify=y, test_size=0.3, random_state=0)
X_train_cat, X_test_cat = train_test_split(X_cat_oh, stratify=y, test_size=0.3, random_state=0)

X_train = np.hstack((X_train_z, X_train_cat))
X_test = np.hstack((X_test_z, X_test_cat))

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

estimator = LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
optimizer.fit(X_train, y_train)

print(optimizer.best_estimator_)
print('X_train (new_auto_balance): ', optimizer.best_score_)
plt.title('X_train (new_auto_balance)')
plot_scores(optimizer)
auc = roc_auc_score(y_test, optimizer.predict_proba(X_test)[:, 1])
print('auc = ', auc)
write_answer_4(auc)

"""Инициализируем класс, который выполняет преобразование"""
transform = PolynomialFeatures(2)
"""Обучаем преобразование на обучающей выборке, применяем его к тестовой"""
example_data_train_poly = transform.fit_transform(example_data_train)
example_data_test_poly = transform.transform(example_data_test)
"""Обращаем внимание на параметр fit_intercept=False"""
optimizer = GridSearchCV(LogisticRegression(class_weight='balanced', fit_intercept=False), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train_poly, example_labels_train)
Z = optimizer.predict(transform.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
plt.title('With class weights')
plt.show()

print('example_data_train_poly.shape = ', example_data_train_poly.shape)


transform = PolynomialFeatures(2)
X_train_z_pol = transform.fit_transform(X_train_z)
X_test_z_pol = transform.transform(X_test_z)

scaler = StandardScaler()

X_train_z_pol_scaled = scaler.fit_transform(X_train_z_pol, y_train)
X_test_z_pol_scaled = scaler.transform(X_test_z_pol)

X_train = np.hstack((X_train_z_pol_scaled, X_train_cat))
X_test = np.hstack((X_test_z_pol_scaled, X_test_cat))

estimator = LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced', fit_intercept=False)
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
optimizer.fit(X_train, y_train)

auc = roc_auc_score(y_test, optimizer.predict_proba(X_test)[:, 1])
print('auc = ', auc)
write_answer_5(auc)



X_train_z, X_test_z, y_train, y_test = train_test_split(X_real_zeros, y, stratify=y, test_size=0.3, random_state=0)
X_train_cat, X_test_cat = train_test_split(X_cat_oh, stratify=y, test_size=0.3, random_state=0)

scaler = StandardScaler()

X_train_real_scaled = scaler.fit_transform(X_train_z, y_train)
X_test_real_scaled = scaler.transform(X_test_z)

X_train = np.hstack((X_train_real_scaled, X_train_cat))
X_test = np.hstack((X_test_real_scaled, X_test_cat))


estimator = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
optimizer = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy')
optimizer.fit(X_train, y_train)
print(optimizer.best_estimator_)

auc = roc_auc_score(y_test, optimizer.predict_proba(X_test)[:, 1])
print('auc = ', auc)
coefs = optimizer.best_estimator_.coef_

print(optimizer.best_estimator_.coef_)
coefs = optimizer.best_estimator_.coef_
print('coefs.shape = ', coefs.shape)
features = []

for i in range(X_real_zeros.shape[1]):
    if coefs[0, i] == 0:
        features.append(i)


print('features = ', features)
write_answer_6(features)
