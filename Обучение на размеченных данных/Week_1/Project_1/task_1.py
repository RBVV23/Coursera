import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


advert_data = pd.read_csv('advertising.csv')
print(advert_data.head())
print(advert_data.info())