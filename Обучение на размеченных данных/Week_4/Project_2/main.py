from sklearn import ensemble, model_selection
import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import xgboost as xgb
print(xgb.__version__)
import matplotlib.pyplot as plt

def write_answer(answer, number):
    name = 'ans{}.txt'.format(number)
    with open(name, "w") as fout:
        fout.write(str(answer))


