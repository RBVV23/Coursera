from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 25)

data = pd.read_csv('credit_card_default_analysis.csv')

print(data.head())
print((data.shape))
