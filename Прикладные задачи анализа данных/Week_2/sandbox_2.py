import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
# import tensorflow as tf
# print(tf.__version__)


my_points = ['A', 'B', 'C']
my_points = ['A', 'B', 'C']
my_points = ['A', 'B', 'C', 'D']

tracks = [['A'],['A']]

# def func(points, tracks=[], track=[]):
#     for i in range(len(points)):
#         for j in range(len(points)):
#             tracks[j].append(points[i])


def func(mass, new_mass=[]):
    for i in range(len(mass)):
        for j in range(len(mass)):
            new_mass.append(mass[i])
    mass.pop(0)
    if len(mass) == 0:
        return new_mass
    return func(mass, new_mass)



res = func(my_points)

print(res)