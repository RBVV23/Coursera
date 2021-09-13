import numpy as np
from sklearn import model_selection, metrics, datasets, linear_model, tree, naive_bayes
from math import ceil
import numpy as np

def my_print_features(dataset, in_line=3):
    feature_names = dataset.feature_names
    n_lines = ceil(len(feature_names)/in_line)
    # print('n_lines = ', n_lines)
    n_last_line = len(feature_names) - in_line*(n_lines-1)
    # print('n_last_line = ', n_last_line)
    print('\nСписок признаков: ')
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
    print()
def my_print_targets(dataset, r=2):
    target_names = dataset.target_names
    print('\nЦелевая переменная принимает значения (метки): ')
    for target in enumerate(target_names):
        print('\t {} ({}): {} раз(-a) - {}%'.format(target[1], target[0], sum(dataset.target == target[0]),
              round(sum(dataset.target == target[0])/len(dataset.target)*100, r)))
    print()

def write_answer(answer, number):
    name = 'ans{}.txt'.format(number)
    with open(name, "w") as fout:
        fout.write(str(answer))

digits = datasets.load_digits()
breast_cancer = datasets.load_breast_cancer()
# my_print_features(digits, 8)
# my_print_targets(digits, 1)

X = digits.data
y = digits.target

classifer = naive_bayes.BernoulliNB()
res = model_selection.cross_val_score(classifer, X, y)
answer_1 = res.mean()
print('answer_1 = ', res)
write_answer(answer_1, 1)




