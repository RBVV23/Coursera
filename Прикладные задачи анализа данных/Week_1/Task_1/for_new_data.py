import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

# скопируйте новые строки таблицы и вставьте в файл filename.txt
# поменяйте расширение с .txt на .csv
# на выходе получится новый файл в формате оригинальной таблицы .csv из задания, но из новых данныч

new_data = pd.read_csv('filename.csv', sep='\t')
print(new_data.columns)

new_data.drop(['WAG_M', 'WAG_M_SA'], axis=1, inplace=True)

# print(new_data.head())
print(new_data.head(-5))
first_date = new_data.month[0]
print('first_date = ', first_date)
# new_data.drop(['month'], axis=1, inplace=True)

date_list = [datetime.datetime.strptime("2016.01.01","%Y.%m.%d") + relativedelta(year=2016+(x)//12, month=1+(x)%12) for x in range(new_data.shape[0])]
print(date_list[:5])
# new_data.index.name = 'month'
new_data.month = list(map(lambda x: x.strftime("%Y.%m.%d"), date_list))
print(new_data)


new_data.to_csv('new_WAG_C_M.csv', sep=';', index=False)

