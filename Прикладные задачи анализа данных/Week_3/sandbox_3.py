import pandas as pd
# print(pd.__version__)
import numpy as np
# print(np.__version__)
import itertools

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))

points = [1, 2, 3,4]

def func(A=[], base=[]):
    base = A
    print('base = ', base)
    print('len(base) = ', len(base))
    print('base[1] = ', base[1])
    for i in range(len(base) * 1):
        # for j in range(len(base)-1):
        A.append(base[i] * 1)
    print(A)
    for i in range(len(A)):
        A[i].append(1 + (i + 1) % 3)
    print(A)
    # return new_tracks

# res = func(points)
# # res[1] = [2, 1, 3, 1]
# print(res)

points = [2,3,4]
# res2 = func(points, res)
# print(res2)


init = list([[]])
# A = [[None]]
# B = A*3
# print(B)
# for m in B:
#     m.append('A')
#     print(m)

# print(res[1])
# points = [2, 3,4]
# res2 = func(points, res)
# print(res2)

# m =3
# n=4
# init =[0]
# print()
# A = [init * 1 for i in range(3)]
# # A = list(A*3)
# for i in range(len(A)):
#     A[i].append(i+1)
# # print(A)
#
# init = A
#
# # A = [ * 1 for i in range(3)]
# # # A = list(A*3)
# # for i in range(len(A)):
# #     A[i].append(i+1)
# # print(A)
#
# init = []
# L = 3
# M = init*L
#
# for i in range(L):
#     M.append(init* 1)
#
# print(M)
#
# for i in range(len(M)):
#     M[i].append(i+1)
#
# print(M)
#
# newM = M*2
#
# for i in range(len(M)):
#     newM[i].append(i+2)
#
# # newM = M*2
# print(newM)
# # newM[0].append(10)
# # print(newM[1])
#
# # init = []
# # L = 3
# # M = init*L
# # for i in range(L):
#     M.append([0]* 1)

points = [1,2,3]
nsb = 3
nsb = len(points)
A = []
base = []
for i in range(nsb):
    A.append(base * 1)
print(A)
for i in range(len(A)):
    A[i].append(0)
print(A)

print('===========================')
for i in range(len(A)):
    A[i].append(i+1)
print(A)



nsb = 2
# m = 4
# A = []
base = A
print('base = ', base)
print('len(base) = ', len(base))
print('base[1] = ', base[1])
for i in range(len(base)*1):
    # for j in range(len(base)-1):
    A.append(base[i] * 1)
print(A)
for i in range(len(A)):
    A[i].append(1+(2 + i + (i+1)//len(base) )%3 )
print(A)


# base = A
# print('base = ', base)
# print('len(base) = ', len(base))
# print('base[1] = ', base[1])


# for i in range(len(base)*1):
#     # for j in range(len(base)-1):
#     A.append(base[i] * 1)
# print(A)
# for i in range(len(A)):
#     A[i].append(1+(i+1)%3)
# print(A)