import pandas as pd
import numpy as np
import collections

print(pd.__version__)
print(np.__version__)


# def my_write_answer(answer, part, number):
#     name = 'answer' + str(part) + str(number) + '.txt'
#     with open(name, 'w') as file:
#         file.write(str(answer))
# def my_precision(d, k):
#     sum = 0
#     for i in d[:k]:
#         sum += i
#     return sum/k
# def my_AP(d):
#     k = len(d)+1
#     sum_1 = 0
#     sum_2 = 0
#     for n,i in enumerate(d):
#         sum_1 += i * my_precision(d,n+1)
#         sum_2 += i
#     return sum_1/sum_2
#
# vec_1 = [1, 0, 0, 1]
# vec_2 = [1, 1, 0, 0]
#
# print(my_AP(vec_1))
# print(my_AP(vec_2))
# answer11 = my_AP(vec_2) - my_AP(vec_1)
# print('answer 1.1 = ', answer11)
#
# mass = ['A','B','C','D','E', 'F', 'G', 'H']
# new_mass = mass
# l = len(mass) - 1
# for i, egg in enumerate(reversed(mass)):
#     print('{}: mass[{}] = {}'.format(i, l-i, egg))
#     if (egg == 'A') or (egg == 'C') or (egg == 'H'):
#         new_mass.pop(l-i)
#
# print(new_mass)
#
# mama = [0,1,2,3,4,5]
# # mama.pop((3,5))
# print(mama)


my_dict = {'23': 10, '10': 5, '8': 3, '99': 1, '7': 1}
my_test = ['8', '99', '8', '10', '7', '23']
my_dict = collections.Counter()
for word in my_test:
    my_dict[word] += 1
print(my_dict)


# array = [[151, 5],
#          [132, 3],
#          [113, 1],
#          [144, 4],
#          [125, 2]]
# def my_sort(session):
uni_session = np.unique(my_test)
array = np.zeros((len(uni_session), 2))
print(uni_session)
