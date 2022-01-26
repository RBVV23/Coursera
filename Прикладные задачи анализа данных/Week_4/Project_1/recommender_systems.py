import pandas as pd
import numpy as np

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))

def my_data_prepare(input_file_names, prefix='my_res'):
    prepared_file_names = []
    for file_name in input_file_names:
        with open(file_name) as f:
            pre_data = f.read().split('\n')

        post_data = pre_data
        print('{}:'.format(file_name))
        print('\tЗаписей ДО обработки - {}'.format(len(pre_data)))
        L = len(pre_data) - 1

        for i, session in enumerate(reversed(pre_data)):
            if session[-1] == ';':
                post_data.pop(L - i)

        print('\tЗаписей ПОСЛЕ обработки - {}'.format(len(post_data)))
        print()

        new_name = prefix + '_' + file_name[9:]
        prepared_file_names.append(new_name)
        with open(new_name, 'w') as f:
            post_data = map(lambda x: x + '\n', post_data)
            f.writelines(post_data)

    return prepared_file_names


# input_file_names = ['coursera_sessions_train.txt', 'coursera_sessions_test.txt']
#
# prepared_file_names = my_data_prepare(input_file_names)
# print(prepared_file_names)

views = dict()
purchases = dict()

df = pd.DataFrame(columns=['VIEWS', 'PURCHASES'])
# data.columns = ['VIEWS', 'PURCHASES']
df.VIEWS = [[1,2], [3,4], [1,2,3,4]]


with open('coursera_sessions_train.txt') as f:
    data = f.read().split('\n')

for string in data[:10]:
    print(string)
    pre = string.split(';')
    view = pre[0].split(',')
    purchase = pre[1].split(',')
    print('\tПросмотрено: {}\n\tКуплено: {}\n'.format(view, purchase))
