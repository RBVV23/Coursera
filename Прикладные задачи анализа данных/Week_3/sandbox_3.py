import pandas as pd
# print(pd.__version__)
import numpy as np
# print(np.__version__)
import itertools

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))

def my_path_choose(points, path=[]):
    all_points = points
    for i in range(len(points)):
        path.append(points[i])
        if len(points.pop(i)):
            my_path_choose(path, points)
        else:
            return path


matr = [[1,2,3],[4,5,6],[7,8,9]]
mass = [[0, 1, 2, 3],[-1, 0, 4, 5], [-2, -4, 0, 6], [-3, -5, -6, 0]]
print(mass)
# print(matr[1][2])


for y in range(len(mass)):
    for x in range(y+1, len(mass[0])):
        print(mass[y][x], end=' ')

# print()
# print(len(matr))
# print(len(matr[0]))
#
# for i in range(len(mass)):
#     print(mass[i])
#
points = ['A','B','C','D']
#
# for p in points:
#     print(f'{p} -> ', end='')
# print()

# print(len(list(itertools.permutations(points, len(points)))))
# print(list(itertools.permutations(points, len(points))))

print(points.pop(3))
# print(points)
points_0 = list(points)

# my_path_choose(points)
# print(len(points))

# L = len(points) - 1 # L = 2
# print('points = ', points)
# for i in range(L+1):
#     print('points_0 = ', points_0)
#     # print(L-i)
#     # print(f'points = {points}')
#     # print(points[L - i], end='->')
#     points_1 = list(points_0)
#     # print(f'points_1 = {points_1}')
#     points_1.pop(L-i)
#     # print(f'points_1 = {points_1}')
#     # print(f'points = {points}')
#     print('\tpoints_1 = ', points_1)
#     for j in range(L):
#         # print(points_1[L-1-j], end='->')
#         # print('\t', points_1[L-1-j])
#         points_2 = list(points_1)
#         points_2.pop(L-j-i)
#         print('\t\tpoints_2 = ', points_2)
#         for k in range(len(points_2)):
#             # print(points_2[k])
#             print(f'{points[L-i]}->{points_1[L-1-j]}->{points_2[L-2-k]}')
#             # print(f'{points[L-i]}->{points_1[L-1-j]}->{points_2[k]}')
#     # print('i =', i)
#     points_0.pop(L-i)

# bag = ['C','B','A']
points = ['A', 'B', 'C']



def func(points, track=[]):
    if len(points)==1:
        return track
    track.append(points[0])
    print(track)
    points.pop(0)
    return func(points, track)






res = func(points)
print(res)

