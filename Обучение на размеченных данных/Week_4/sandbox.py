import pandas as pd
# print(pd.__version__)
import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt



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

