import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import re

# скопируйте новые строки таблицы и вставьте в файл filename.txt
# поменяйте расширение с .txt на .csv
# на выходе получится новый файл в формате оригинальной таблицы .csv из задания, но из новых данныч

new_data = pd.read_csv('filename.csv', sep='\t')
print(new_data.columns)

new_data.drop(['WAG_M', 'WAG_M_SA'], axis=1, inplace=True)

# print(new_data.head())
# new_data.columns = ['WAG_M', 'WAG_M_SA', 'WAG_C_M']
print(new_data)
first_date = new_data.month[0]
print('first_date = ', first_date)
# new_data.drop(['month'], axis=1, inplace=True)
#
date_list = [datetime.datetime.strptime("1993.01.01","%Y.%m.%d") + relativedelta(year=1993+(x)//12, month=1+(x)%12) for x in range(new_data.shape[0])]
# print(date_list[:5])
# # new_data.index.name = 'month'
new_data.month = list(map(lambda x: x.strftime("%Y.%m.%d"), date_list))
new_data['WAG_C_M'] = list(map(lambda x: re.sub(',', '.', str(x)), new_data['WAG_C_M']))
print('new_data')
print(new_data)
#
#
#
# new_data.to_csv('new_WAG_C_M.csv', sep=';', index=False)
#
# old_data = pd.read_csv('WAG_C_M.csv', sep=';', index_col=['month'], parse_dates=['month'],
#                    dayfirst=True)
#
# # old_data['month'] = old_data.index
# old_data.insert(loc=0, column='month', value=old_data.index)
#
#
# old_data.index=list(range(old_data.shape[0]))
# print('old_data')
# print(old_data)
#
#
#
#
# updated_data = old_data.append(new_data)
#
# updated_data.index=list(range(updated_data.shape[0]))
# print('updated_data')
# print(updated_data)
#
#
new_data.to_csv('updated_WAG_C_M.csv', sep=';', index=False)
#
