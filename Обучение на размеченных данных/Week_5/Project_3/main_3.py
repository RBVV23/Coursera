import numpy as np
from sklearn import model_selection, metrics, datasets, linear_model, tree

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError
from sklearn.model_selection import train_test_split

def my_euclid_dist(obj_1, obj_2):
    if len(obj_1) == len(obj_2):
        L = len(obj_1)
    else:
        raise IndexError('Объекты имеют разную размерность')
        return
    dist = 0
    for i in range(L):
        dist += (obj_1[i] - obj_2[i])**2
    dist = dist**0.5
    return dist



def my_1NN(obj, X, y):
    N = X.shape[0]
    matrix = np.ones((N,2))*-1
    for i in range(N):
        matrix[i][0] = my_euclid_dist(obj, X[i])
        matrix[i][1] = y[i]
    ind = matrix[:,0].argmin()
    return int(matrix[ind,1])




digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

y_predict = []
for obj in X_test:
    prediction = my_1NN(obj, X_train, y_train)
    y_predict.append(prediction)
print('y_test: ', y_test)
print('y_predict: ', y_predict)



