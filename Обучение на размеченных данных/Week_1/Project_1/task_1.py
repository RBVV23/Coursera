import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



def mserror(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean((y - y_pred)**2)

def normal_equation(X, y):
    X_t = X.transpose()
    X_obr = np.dot(X_t, X)
    X_obr = np.linalg.inv(X_obr)
    Sol = np.dot(X_obr, X_t)
    return np.dot(Sol,y)

def linear_prediction(X, w):
    return np.dot(X,w)


def gradient_step(X, y, w, eta=0.01):
    L = X.shape[0]
    grad0 = 2. / L * np.sum((np.sum(X * w, axis=1) - y) * X[:, 0], axis=0)
    grad1 = 2. / L * np.sum((np.sum(X * w, axis=1) - y) * X[:, 1], axis=0)
    grad2 = 2. / L * np.sum((np.sum(X * w, axis=1) - y) * X[:, 2], axis=0)
    grad3 = 2. / L * np.sum((np.sum(X * w, axis=1) - y) * X[:, 3], axis=0)
    return w - eta * np.array([grad0, grad1, grad2, grad3])

def gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                     min_weight_dist=1e-8, verbose=False):
    weight_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0
    while weight_dist > min_weight_dist and iter_num < max_iter:
        w_new = gradient_step(X=X, y=y, w=w, eta=eta)
        weight_dist = (sum((w - w_new) ** 2)) ** 0.5
        y_pred = linear_prediction(X, w_new)
        error = mserror(y, y_pred)
        errors.append(error)
        w = w_new
        iter_num += 1
        if (iter_num % 100) == 0 and verbose == True:
            print('iter_num = ', iter_num)
            print('\tweight_dist = ', weight_dist)
            print('\terror = ', error)
    return w, errors


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    res = X[train_ind][0]*w[0] + X[train_ind][1]*w[1] + X[train_ind][2]*w[2] + X[train_ind][3]*w[3]
    grad0 =  ( res - y[train_ind] ) * X[train_ind][0]
    grad1 =  ( res - y[train_ind] ) * X[train_ind][1]
    grad2 =  ( res - y[train_ind] ) * X[train_ind][2]
    grad3 =  ( res - y[train_ind] ) * X[train_ind][3]
    return  w - 2*eta * np.array([grad0, grad1, grad2, grad3])

def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    weight_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0
    np.random.seed(seed)
    while weight_dist > min_weight_dist and iter_num < max_iter:
        iter_num += 1
        random_ind = np.random.randint(X.shape[0])
        w_new = stochastic_gradient_step(X=X, y=y, w=w, train_ind=random_ind, eta=eta)
        weight_dist = (sum((w - w_new) ** 2)) ** 0.5
        y_pred = linear_prediction(X, w_new)
        error = mserror(y, y_pred)
        errors.append(error)
        w = w_new
        if (iter_num % 100) == 0 and verbose == True:
            print('iter_num = ', iter_num)
            print('\tweight_dist = ', weight_dist)
            print('\terror = ', error)
            print('\trandom_ind = ', random_ind)
    if verbose == True:
        print('w = ', w)
        print('errors[0] = ', errors[0])
        print('errors[-1] = ', errors[-1])
        print('mean.errors[-1] = ', np.mean(errors))
    return w, errors


adver_data = pd.read_csv('advertising.csv')
adver_data.head()
adver_data.info()
# adver_data = adver_data.head()

X = np.array([adver_data['TV'], adver_data['Radio'], adver_data['Newspaper']])
X = X.T
y = np.array(adver_data['Sales'])
# print(X)

means = np.mean(X, axis=0)
X = X - means
stds = np.std(X, axis=0)
X = X /stds
# print(X)

N = X.shape[0]
print(N)
one = np.ones((N,1))
X = np.hstack((one, X))
# print(X)

med = np.median(np.array(adver_data['Sales']))
y_pred = np.ones((N))*med
y = np.array(adver_data['Sales'])
# print(y_pred, y)
answer1 =  mserror(y, y_pred)
print('\tanswer 1 = ', round(answer1, 3))
# answer1 = 28.346

norm_eq_weights = normal_equation(X, y)
print('norm_eq_weights = ', norm_eq_weights)
# [ 14.0225 3.91925365  2.79206274 -0.02253861 ]

X_0 = np.array([1, 0, 0, 0])
answer2 = np.dot(X_0,norm_eq_weights)
print('\tanswer 2 = ', np.round(answer2, 3))
# answer 2 = 14.022

y_pred = linear_prediction(X, norm_eq_weights)
y = np.array(adver_data['Sales'])

answer3 = mserror(y, y_pred) # Ваш код здесь
print('\tanswer 3 = ', round(answer3, 3))
# answer 3 = 2.784

w_init = np.array([0, 0, 0, 0])
stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, w_init, eta=0.01, max_iter=1e5, min_weight_dist=1e-8,
                            seed=42, verbose=False)

plt.plot(range(50), stoch_errors_by_iter[:50])
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.show()

plt.plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
plt.xlabel('Iteration number')
plt.ylabel('MSE')

print('stoch_grad_desc_weights = ', stoch_grad_desc_weights)
print('stoch_errors_by_iter[-1] = ', stoch_errors_by_iter[-1])

stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e5,
                                min_weight_dist=1e-8, seed=42, verbose=False)

# stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, w_init, eta=1e-2/200, max_iter=1e5,
#                                 min_weight_dist=1e-8, seed=42, verbose=False)

y_pred = linear_prediction(X, stoch_grad_desc_weights)
y = np.array(adver_data['Sales'])
answer4 = mserror(y, y_pred)
print('\tanswer 4 = ', round(answer4, 3))
# answer 4 = 2.714 (для получения такого ответа нужно уменьшить eta в 200 раз)