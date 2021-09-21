import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn

data = pandas.read_csv("train.csv", na_values="NaN")
print(data.head())