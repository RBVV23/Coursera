import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
# import tensorflow as tf
# print(tf.__version__)


points = ['A', 'B', 'C']
points1 = list(points)
track = []

def func(points, base=[]):
    # размножение
    A = []
    for i in range(len(points)):
        A.append(base * 1)
    # заполнение
    for i in range(len(points)):
        A[i].append(points[i])
    return A

points = [1, 2, 3]
# res = func(points)
# print(res)

# points = [2, 3]
# print(func(points), res[0])

# nsb = 3
# nsb = len(points)
# A = []
# base = []
# for i in range(nsb):
#     A.append(base * 1)
# print(A)

# размножение

Lzap = 3
points = [1,2,3]
base = [[0]]
l = len(base[0])
L = len(points) - len(base[0])
print('L = ', L)
print('l = ', l)
print(base)
A = [[0]]

for i in range(L):
    for b in base:
        A.append(b * 1)
print(A)
# заполнение
# for i in range(Lzap):
#     A[i].append(points[i])
for i, a in enumerate(A):
    a.append(1 + (0+i - (i//3) )%3)
    # a.append( (1+i)//l )
print(A)

print('===================')


#
# points = [1,2,3]
# Lzap = 9
base = [[0, 1], [0, 2], [0, 3]]

print('len(base[0]) = ', len(base[0]))
l = len(base[0])
L = len(points) - len(base[0])
print('L = ', L)
print('l = ', l)
print(base)
# # A = []
#
for i in range(L):
    for b in base:
        A.append(b * 1)
print(A)

for i, a in enumerate(A):
    a.append(1 + (1+i + (i//3) )%3)
print(A)

#
print('===============================================================')
#
base = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2], [0, 2, 3], [0, 3, 1]]


print('len(base[0]) = ', len(base[0]))
l = len(base[0])
L = len(points) - len(base[0])
print('L = ', L)
print('l = ', l)
print(base)
# # A = []
#
for i in range(L):
    for b in base:
        A.append(b * 1)
print(A)

for i, a in enumerate(A):
    a.append(1 + (2+i - (i//3) )%3)
print(A)




# # A = []
# for b in base:
#     for i in range(len(points)-1):
#         A.append(b * 1)
# print(A)
# # заполнение
# for i in range(Lzap):
#     A[i].append(points[i%3])
# print(A)