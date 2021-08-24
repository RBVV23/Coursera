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

def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
  #     j = np.randint(0, N-1, 1)
#     print('X[train_ind][0] = ', X[train_ind][0])
#     print('X[train_ind][1] = ', X[train_ind][1])
#     print('X[train_ind][2] = ', X[train_ind][2])
#     print('X[train_ind][3] = ', X[train_ind][3])
#     print('y[train_ind] = ', y[train_ind])
#     print('w[0] = ', w[0])
#     print('w[1] = ', w[1])
#     print('w[2] = ', w[2])
#     print('w[3] = ', w[3])
    res = X[train_ind][0]*w[0] + X[train_ind][1]*w[1] + X[train_ind][2]*w[2] + X[train_ind][3]*w[3]
    grad0 = 2 * ( res - y[train_ind] ) * X[train_ind][0] # Ваш код здесь
    grad1 = 2 * ( res - y[train_ind] ) * X[train_ind][1] # Ваш код здесь
    grad2 = 2 * ( res - y[train_ind] ) * X[train_ind][2] # Ваш код здесь
    grad3 = 2 * ( res - y[train_ind] ) * X[train_ind][3] # Ваш код здесь
    return  w - eta * np.array([grad0, grad1, grad2, grad3])

def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом.
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)

    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])
        #         print('random_ind =', random_ind)
        # Ваш код здесь
        w_new = stochastic_gradient_step(X=X, y=y, w=w, train_ind=random_ind, eta=eta)
        #         print('dw=',dw)
        #         print('w=', w)
        #         print('w.shape[1]=', w.shape[0])
        #         print('(w - w_new)^2=',sum((w - w_new))**2)
        weight_dist = sum((w - w_new) ** 2) / (w.shape[0])
        y_pred = linear_prediction(X, w_new)
        error = mserror(y, y_pred)
        errors.append(error)
        #         errors.append(mserror(y, y_pred))
        w = w_new
        iter_num += 1
        if (iter_num % 100) == 0 and verbose == True:
            print('iter_num = ', iter_num)
            print('\tweight_dist = ', weight_dist)
            print('\terror = ', error)
    return w, errors




adver_data = pd.read_csv('advertising.csv')
print(adver_data.head())
print(adver_data.info())

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
# print(N)
one = np.ones((N,1))
X = np.hstack((X, one))
# print(X)

med = np.median(np.array(adver_data['Sales']))
y_pred = np.ones((N))*med
y = np.array(adver_data['Sales'])
# print(y_pred, y)
answer1 =  mserror(y, y_pred)
print('answer1 = ', round(answer1, 3))
# answer1 = 28.346

norm_eq_weights = normal_equation(X, y)
print(norm_eq_weights)

X_0 = np.array([0, 0, 0, 1])
answer2 = np.dot(X_0, norm_eq_weights)

# print(answer2.shape)
print('answer2 = ', np.round(answer2, 3))

# answer2 = 14.022

y_pred = linear_prediction(X, norm_eq_weights)
y = np.array(adver_data['Sales'])

answer3 = mserror(y, y_pred)
print('answer3 = ', round(answer3, 3))

# answer3 = 2.784

w_init = np.array([0, 0, 0, 0])

w, errors =stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4, min_weight_dist=1e-8,
                            seed=42, verbose=True)
print('w = ', w)
print('errors[0] = ', errors[0])
print('errors[-1] = ', errors[-1])
print('mean.errors[-1] = ', np.mean(errors))

stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e5, min_weight_dist=1e-8,
                            seed=42)

plt.plot(range(50), stoch_errors_by_iter[:50])
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.show()

plt.plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.show()

print('stoch_grad_desc_weights = ', stoch_grad_desc_weights)
print('stoch_errors_by_iter[-1]', stoch_errors_by_iter[-1])

y_pred = linear_prediction(X, stoch_grad_desc_weights)
y = np.array(adver_data['Sales'])
answer4 =  mserror(y, y_pred)
print('answer4 = ', round(answer4, 3))
# answer4 = 2.856