import pandas as pd
import numpy as np

# def my_pandas_info()

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 100)


data = pd.read_csv('checkins.csv', sep='|', header=0, skipinitialspace=True)
columns = data.columns
print('data.shape = ', data.shape)
data = data.drop(data.index[[0, -1]])


my_columns = []
for col in columns:
    new_name = col.strip()
    my_columns.append(new_name)
    data = data.rename(columns={col: new_name})

data = data.astype({'id': int, 'user_id': int, 'venue_id': int})

print(data.head())
# print(data.info())

for iter in data['latitude']:
        if iter.isna():
            print(iter)