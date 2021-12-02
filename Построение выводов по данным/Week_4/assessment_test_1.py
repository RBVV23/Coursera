from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools


data = pd.read_csv('ab_browser_test.csv')

print(data.head())
print(data.info())