import pandas as pd
import numpy as np
import collections

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

views = []
purchases = []
view_dict = collections.Counter()
purchase_dict = collections.Counter()



# df.VIEWS = [[1,2], [3,4], [1,2,3,4]]


with open('coursera_sessions_train.txt') as f:
    data = f.read().split('\n')

for string in data:
    # print(string)
    pre = string.split(';')
    # view = list(map(lambda x: int(x), pre[0].split(',')))
    # purchase = list(map(lambda x: int(x), pre[1].split(',')))
    view = pre[0].split(',')
    purchase = pre[1].split(',')
    for id in view:
        view_dict[id] += 1
    views.append(view)
    for id in purchase:
        purchase_dict[id] += 1
    purchases.append(purchase)
    # print('\tПросмотрено: {}\n\tКуплено: {}\n'.format(view, purchase))

mass = [views, purchases]
df = pd.DataFrame(mass).T
df.columns = ['VIEWS', 'PURCHASES']
# print(df)

print('view_dict:')
# print(view_dict, '\n')
# print()
print('purchase_dict:')
purchase_dict.pop('')
# print(purchase_dict)
# print(data[10])
# print(df.loc[10].VIEWS)
# print(df.loc[10].PURCHASES)

# for session in range(df.shape[0]//10000):
for session in range(6,8):
    print(df.loc[session])
    views = df.loc[session].VIEWS
    purchases = df.loc[session].PURCHASES
    print('\tviews :', views)
    print('\tpurchases: ', purchases)

# print(df.loc[10])
# print(df.shape[0])

my_dict = {'23': 10, '10': 5, '8': 3, '7': 1}
my_test = ['8', '7','8', '10', '99', '23', '99']
# def my_sort(session):
uni_session = np.unique(my_test)
print(uni_session)