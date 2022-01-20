import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))

def my_precision(d):
    sum = 0
    for i in d:
        sum += i
    return sum/len(d)

def my_AP(d):
    k = len(d)+1
    sum_1 = 0
    sum_2 = 0
    for n,i in enumerate(d):
        sum_1 += i * i/(n+1)
        sum_2 += i
    return sum_1/sum_2

vec_1 = [1, 0, 0, 1]
vec_2 = [1, 1, 0, 0]

print(my_AP(vec_1))
print(my_AP(vec_2))
answer11 = my_AP(vec_2) - my_AP(vec_1)
print('answer 1.1 = ', answer11)