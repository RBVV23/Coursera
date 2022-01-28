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
points_0 = points

# my_path_choose(points)
# print(len(points))

L = len(points) - 1 # L = 2
for i in range(L+1):
    # print(L-i)
    # print(f'points = {points}')
    # print(points[L - i], end='->')
    points_1 = list(points)
    # print(f'points_1 = {points_1}')
    points_1.pop(L-i)
    # print(f'points_1 = {points_1}')
    # print(f'points = {points}')
    for j in range(L):
        # print(points_1[L-1-j], end='->')
        # print('\t', points_1[L-1-j])
        points_2 = list(points_1)
        points_2.pop(L-1-j)
        for k in reversed(range(len(points_2))):
            # print(points_2[k])
            print(f'{points[L-i]}->{points_1[L-1-j]}->{points_2[L-2-k]}')
            # print(f'{points[L-i]}->{points_1[L-1-j]}->{points_2[k]}')

bag = ['A','B','C']
bag_1 = ['A','B','C']
bag_2 = ['A','B','C']
bag_3 = ['A','B','C']

for i in range(len(bag_1)):
    print(bag_1[i])
    bag_2 = list(bag_1)
    bag_2.pop(i)
    for j in range(len(bag_2)):
        bag_3 = list(bag_2)
        bag_3.pop(j)
        print('\t', bag_2[j])
        for k in range(len(bag_3)):
            print('\t\t', bag_3[k])
    bag.pop(i)
