from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv('bikes_rent.csv')
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 15)
print(data.head())
print(data.info())

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))
for idx, feature in enumerate(data.columns[:-1]):
    # print([idx // 4, idx % 4])
    data.plot(feature, "cnt", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
# plt.show()

print('\n======================== МАТРИЦА КОРРЕЛЯЦИЙ ПИРСОНА ========================')
print(data.corr())
print('\t ========================================================================================================\n')


print('\n======================== СРЕДНИЕ ЗНАЧЕНИЯ ПО ПРИЗНАКАМ ========================')
print(np.mean(data, axis=0))
print('\t ========================================================================================================\n')

data_shuffled = shuffle(data, random_state=123)
X = scale(data_shuffled[data_shuffled.columns[:-1]])
y = data_shuffled["cnt"]

regressor = LinearRegression()
regressor.fit(X, y)
for col, cof in zip(data.columns, regressor.coef_):
    print('{}: {:.3f}'.format(col, cof))

l1_regressor = Lasso(random_state=123)
l1_regressor.fit(X, y)
for col, cof in zip(data.columns, l1_regressor.coef_):
    print('{}: {:.3f}'.format(col, cof))

l2_regressor = Ridge(random_state=123)
l2_regressor.fit(X, y)
for col, cof in zip(data.columns, l2_regressor.coef_):
    print('{}: {:.3f}'.format(col, cof))

alphas = np.arange(1, 500, 50)
coefs_lasso = np.zeros((alphas.shape[0], X.shape[1]))
coefs_ridge = np.zeros((alphas.shape[0], X.shape[1]))
for a in enumerate(alphas):
    reg_r = Ridge(random_state=123, alpha=a[1])
    reg_r.fit(X, y)
    coefs_ridge[a[0],:]=reg_r.coef_
    reg_l = Lasso(random_state=123, alpha=a[1])
    reg_l.fit(X, y)
    coefs_lasso[a[0],:]=reg_l.coef_

plt.figure(figsize=(8, 5))
for coef, feature in zip(coefs_lasso.T, data.columns):
    plt.plot(alphas, coef, label=feature, color=np.random.rand(3))
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 0.95))
plt.xlabel("alpha")
plt.ylabel("feature weight")
plt.title("Lasso")
plt.show()

plt.figure(figsize=(8, 5))
for coef, feature in zip(coefs_ridge.T, data.columns):
    plt.plot(alphas, coef, label=feature, color=np.random.rand(3))
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 0.95))
plt.xlabel("alpha")
plt.ylabel("feature weight")
plt.title("Ridge")
plt.show()

alphas = np.arange(1, 100, 5)
new_regressor = LassoCV(cv=3, alphas=alphas)
new_regressor.fit(X, y)
print('alpha: ', new_regressor.alpha_)
for col, cof in zip(data.columns, new_regressor.coef_):
    print('{}: {:.3f}'.format(col, cof))

plt.plot(new_regressor.alphas_, np.mean(new_regressor.mse_path_, axis=1))
plt.show()

for n in range(3):
    plt.figure(figsize=(8, 5))
    minMSE = np.min(new_regressor.mse_path_[:,n])
    alpha_n = np.argmin(new_regressor.mse_path_[:,n])
    alpha_1 = new_regressor.alphas_[alpha_n]
    my_label = 'min(MSE) = {}, alpha = {}'.format(minMSE, alpha_1)
    plt.plot(new_regressor.alphas_, new_regressor.mse_path_[:,n], label=my_label)
    plt.legend(loc="upper right")
    plt.show()