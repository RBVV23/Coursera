import pandas as pd
# print(pd.__version__)
import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt

def get_meshgrid(data, step=0.05, border=5):
    x_min = data[:, 0].min() - border
    x_max = data[:, 0].max() + border
    y_min = data[:, 1].min() - border
    y_max = data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

A = np.array([[1,2],[3,4], [5,6]])
B = np.array([1,2,3])
C = np.array([4,5,6])
# print(A.reshape(1))

X = []
B=B.reshape(3,1)
C=C.reshape(3,1)
# print(B)
# print(np.c_[B, C])

data=A

xx, yy = get_meshgrid(data, step=0.5, border=0.5)
print(A[:,1])

xx, yy = get_meshgrid(train_data)
mesh_predictions = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)