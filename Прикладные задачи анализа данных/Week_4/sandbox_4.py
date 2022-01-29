import pandas as pd
import numpy as np

print(pd.__version__)
print(np.__version__)


def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))
def my_precision(d, k):
    sum = 0
    for i in d[:k]:
        sum += i
    return sum/k
def my_AP(d):
    k = len(d)+1
    sum_1 = 0
    sum_2 = 0
    for n,i in enumerate(d):
        sum_1 += i * my_precision(d,n+1)
        sum_2 += i
    return sum_1/sum_2


vec_1 = [1, 0, 0, 1]
vec_2 = [1, 1, 0, 0]

print(my_AP(vec_1))
print(my_AP(vec_2))
answer11 = my_AP(vec_2) - my_AP(vec_1)
print('answer 1.1 = ', answer11)


mass = ['A','B','C','D','E', 'F', 'G', 'H']
new_mass = mass
l = len(mass) - 1
for i, egg in enumerate(reversed(mass)):
    print('{}: mass[{}] = {}'.format(i, l-i, egg))
    if (egg == 'A') or (egg == 'C') or (egg == 'H'):
        new_mass.pop(l-i)

print(new_mass)


mama = [0,1,2,3,4,5]
# mama.pop((3,5))
print(mama)




# [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 1, 3, 2], [0, 2, 1, 3], [0, 3, 2, 1]]
for i in range(24):
    # n1 = 1 + (0+i + (i//3) )%3
    n1 = 1 + (0+i + 0* (i //4) )%4
    n2 = 1 + (1+i + 1* (i //4) )%4
    n3 = 1 + (2+i + 2* (i //4) )%4
    n4 = 1 + (3+i + 3* (i //4) )%4
    print(f'{n1}{n2}{n3}{n4}')

for i in range(6):
    n1 = 1 + (0+i + 0*(i//3) )%3
    n2 = 1 + (1+i + 1*(i//3) )%3
    n3 = 1 + (2+i + 2*(i//3) )%3
    print(f'{n1}{n2}{n3}')