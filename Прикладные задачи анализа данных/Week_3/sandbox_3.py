import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))