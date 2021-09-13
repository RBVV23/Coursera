import numpy as np
from sklearn import model_selection, metrics, datasets, linear_model, tree
from math import ceil
import numpy as np

def my_print_features(dataset, in_line=3):
    feature_names = dataset.feature_names
    n_lines = ceil(len(feature_names)/in_line)
    # print('n_lines = ', n_lines)
    n_last_line = len(feature_names) - in_line*(n_lines-1)
    # print('n_last_line = ', n_last_line)
    print('Список признаков: ')
    for n in range(n_lines-1):
        line = ''
        for i in range(in_line):
            ind = n*in_line + i
            line += '\t' + str(feature_names[ind])
        print(line)
    line = ''
    n += 1
    for i in range(n_last_line):
        ind = n * in_line + i
        line += '\t' + str(feature_names[ind])
    print(line)

digits = datasets.load_digits()
# my_print_features(digits, 8)

def my_print_targets(dataset, classification=True):
    target_names = dataset.target_names
    print('Целевая переменная принимает значения (метки): ')
    for target in enumerate(target_names):
        print('\t {} ({}) - {} раз(-a)'.format(target[1], target[0], sum(dataset.target == target[0])))





breast_cancer = datasets.load_breast_cancer()
my_print_targets(digits)
# print(sum(breast_cancer.target == 0))

#357 ==1 - benign