import pandas as pd
import numpy as np

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))