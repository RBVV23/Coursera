import pandas as pd
# pd.__version__
import seaborn as sns
# sns.__version__
import numpy as np
# np.__version__
import matplotlib.pyplot as plt
# plt.__version__

# data = pd.read_csv('weights_heights.csv')
data = pd.read_csv('weights_heights.csv',  index_col='Index')

print(data.head())
# print(data.shape)
# print(data.info())

data.plot(y='Height', kind='hist', color='red', title='Распределние роста подростков (дюймы)')
plt.show()