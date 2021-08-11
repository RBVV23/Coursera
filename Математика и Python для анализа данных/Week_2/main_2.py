# Знакомство с библиотеками Pandas, Numpy

import pandas as pd
import numpy as np

## print(pd.__version__)
print(np.__version__)

# frame = pd.DataFrame( {'numbers': range(10), 'chars': ['a']*10} )
# print(frame)

# frame = pd.read_csv('dataset.tsv')
# print(frame)
frame = pd.read_csv('dataset.tsv', header=0, sep='\t')
print(frame)
print(frame.columns)
print(frame.shape)

new_line = {'Name': 'Perov', 'Birth': '22.03.1990', 'City': 'Penza'}

# frame.append(new_line, ignore_index=True)
frame = frame.append(new_line, ignore_index=True)
print(frame)

frame['IsStudent'] = [False]*5 + [True]*2
print(frame)
# frame = frame.drop([6,5], axis=0)
frame.drop([6, 5], axis=0, inplace=True)
print(frame)
frame.drop('IsStudent', axis=1, inplace=True)
print(frame)

frame.to_csv('dataset_upd.csv', sep=',', header=True, index=None)

frame = pd.read_csv('dataset_upd.csv', header=0, sep=',')
print(frame)
print(frame.dtypes)

frame.Birth = frame.Birth.apply(pd.to_datetime)
print(frame)
print(frame.dtypes)

frame.info()
frame.fillna('разнорабочий', inplace=True)

print(frame['Position'])

# print(frame.Position)
## print(type(frame))

print(frame[['Name', 'Position']])

# print(frame.head(3))
# print(frame[-3:])

# print(frame.loc[[1, 3, 4], ['Name', 'City']])
print(frame.iloc[[1, 3, 4], [0, 2]])

# print(frame[frame.Birth >= pd.datetime(1985,1,1)])
# print(frame[frame['Birth'] >= pd.datetime(1985,1,1)])


# print(frame[(frame['Birth'] >= pd.datetime(1985,1,1)) & (frame.City != 'Москва')])
# print(frame[(frame.Birth >= pd.datetime(1985,1,1)) | (frame['City'] == 'Волгоград')])