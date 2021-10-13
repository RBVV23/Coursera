import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('verizon.txt', sep='\t')
print('data.shape = ', data.shape)
print(data.head())

print(data.Group.value_counts())

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(data[data['Group'] == 'ILEC'].Time, bins=20, color='b', range=(0,100), label='ILEC')
plt.figure(figsize=(12,5))
plt.subplot(1,2,2)
plt.hist(data[data['Group'] == 'CLEC'].Time, bins=20, color='r', range=(0,100), label='CLEC')
plt.show()