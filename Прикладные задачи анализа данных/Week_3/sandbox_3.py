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

print()
print(len(matr))
print(len(matr[0]))

for i in range(len(mass)):
    print(mass[i])

points = ['A','B','C','D']

for p in points:
    print(f'{p} -> ', end='')
print()

print(len(list(itertools.permutations(points, len(points)))))
print(list(itertools.permutations(points, len(points))))

print(points.pop(3))
print(points)
points_0 = points

# my_path_choose(points)

for i in range(len(points_0)):
    print(points_0[i])
    points = points.pop(i)
    # for j in range(len(points)):
    #     print('\t' + points[j])
        # points_3 = points_2.pop(j)
        # for k in range(len(points_3)):
        #     print('\t\t' + points_3[k])