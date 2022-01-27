import pandas as pd
# print(pd.__version__)
import numpy as np
# print(np.__version__)

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))


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