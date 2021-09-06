import pandas as pd
# print(pd.__version__)
import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt

def jinny(p_array):
    sum = 0
    for p in p_array:
        sum+= p*(1-p)
    return sum



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


print(A[:,1])

p1=0.9
p2=0.07
p3=0.03
p_array = np.array([p1, p2, p3])

print(jinny(p_array))