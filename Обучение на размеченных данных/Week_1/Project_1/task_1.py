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

print(np.random.randint(0, 199, 1))
