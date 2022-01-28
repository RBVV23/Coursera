import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
# import tensorflow as tf
# print(tf.__version__)



my_set = {'A','B','C'}

print(my_set)
print(my_set)
my_set_0 = my_set.copy()
L = len(my_set)
for l0 in my_set_0:
    print(l0)
    my_set_1=my_set.copy()
    my_set_1.remove(l0)
    for l1 in my_set_1:
        print('\t', l1)
        my_set_2 = my_set.copy()
        my_set_2.remove(l0)
        my_set_2.remove(l1)
        for l2 in my_set_2:
            print('\t\t', l2)
    my_set.remove(l0)
    # print('\t', my_set_2)

my_seta = {'A','B','C'}
my_setb = {'B','C'}
print(my_seta-my_setb)