import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# def mserror(y, y_pred):
#     y = np.array(y)
#     y_pred = np.array(y_pred)
#     return np.mean((y - y_pred)**2)

def normal_equation(X, y):
    X_t = X.transpose()
    X_obr = np.dot(X_t, X)
    X_obr = np.linalg.inv(X_obr)
    Sol = np.dot(X_obr, X_t)
    return np.dot(Sol,y)  # Ваш код здесь

def linear_prediction(X, w):
    return np.dot(X,w)


# adver_data = pd.read_csv('advertising.csv')
# print(adver_data.head())
# print(adver_data.info())
#
# adver_data = adver_data.head()
# X = np.array([adver_data['TV'], adver_data['Radio'], adver_data['Newspaper']]) # Ваш код здесь
# X = X.T
# y = np.array(adver_data['Sales']) # Ваш код здесь
# # print(X)
#
# means = np.mean(X, axis=0)
# # print(means)
# X = X - means
# stds = np.std(X, axis=0) # Ваш код здесь
# # print(stds)
# X = X /stds
#
# y = np.array(adver_data['Sales'])
# N = X.shape[0]
# med = np.median(np.array(adver_data['Sales']))
# y_pred = np.ones((N))*med
#
#
#






# X * w = y
# w = X-1 * y




# print(normal_equation(X, y))

def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
  #     j = np.randint(0, N-1, 1)
    grad0 = 2 * (X[train_ind][0]*w[0] - y[train_ind]) * X[train_ind][0] # Ваш код здесь
    grad1 = 2 * (X[train_ind][1]*w[1] - y[train_ind]) * X[train_ind][1] # Ваш код здесь
    grad2 = 2 * (X[train_ind][2]*w[2] - y[train_ind]) * X[train_ind][2] # Ваш код здесь
    grad3 = 2 * (X[train_ind][3]*w[3] - y[train_ind]) * X[train_ind][3] # Ваш код здесь
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
        # Ваш код здесь
        dw = stochastic_gradient_step(X, y, w, train_ind=random_ind, eta=eta)
        w_new = w - eta * dw
        weight_dist = ((w - w_new) ^ 2) / (w.shape[1])
        y = y[random_ind]
        y_pred = linear_prediction(X, new_w)
        errors.append(mserror(y, y_pred))
    return w, errors


w_init = np.array([0, 0, 0, 0])
print( stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4, min_weight_dist=1e-8,
                            seed=42, verbose=False) )