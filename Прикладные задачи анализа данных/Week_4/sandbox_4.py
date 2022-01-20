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
    sum_1 = 0
    sum_2 = 0
    for n,i in enumerate(d):
        sum_1 += i*i/n
        sum_2 += i

# vec_1