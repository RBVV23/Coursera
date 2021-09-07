import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import model_selection, datasets, tree, ensemble

def write_answer(answer, number):
    name = 'ans{}.txt'.format(number)
    with open(name, "w") as fout:
        fout.write(str(answer))


digits = datasets.load_digits()
# print(digits.DESCR)

X = digits.data
y = digits.target
#
# print(X.shape)
# print(y.shape)

classifer = tree.DecisionTreeClassifier()
res = model_selection.cross_val_score(classifer, X, y, cv=10)

answer_1 = res.mean()
print('answer_1 = ', answer_1)
write_answer(answer_1, 1)

classifer = ensemble.BaggingClassifier(classifer, n_estimators=100)
res = model_selection.cross_val_score(classifer, X, y, cv=10)
answer_2 = res.mean()
print('answer_2 = ', answer_2)
write_answer(answer_2, 2)
#
d = int((X.shape[0])**0.5)
# print(d)
classifer = ensemble.BaggingClassifier(classifer, max_features=d, n_estimators=100)
res = model_selection.cross_val_score(classifer, X, y, cv=10)
answer_3 = res.mean()
print('answer_3 = ', answer_3)
write_answer(answer_3, 3)


d = int((X.shape[0])**0.5)
classifer = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_features='sqrt'), n_estimators=100)
res = model_selection.cross_val_score(classifer, X, y, cv=10)
answer_4 = res.mean()
# print('n_features = ', classifer.n_features_)
print('answer_4 = ', answer_4)
write_answer(answer_4, 4)

n_est_list = [5, 10, 15, 30, 50, 75, 100, 200, 300, 500]
n_est_y = []
for n_est in n_est_list:
    classifer = ensemble.RandomForestClassifier(n_estimators=n_est)
    res = model_selection.cross_val_score(classifer, X, y, cv=5)
    n_est_y.append(res.mean())
plt.figure()
plt.grid(True)
plt.plot(n_est_list, n_est_y, 'g-', marker='o')
plt.plot(n_est_list[:4], n_est_y[:4], 'r--', marker='o', label='Малое число деревьев')
plt.title('Зависимость качества от числа деревьев')
plt.legend(loc='lower right')
# plt.show()

m_features_list = [5, 10, 40, 50, 60]
m_feat_y = []
for m_feat in m_features_list:
    classifer = ensemble.RandomForestClassifier(n_estimators=100, max_features=m_feat)
    res = model_selection.cross_val_score(classifer, X, y, cv=5)
    m_feat_y.append(res.mean())
plt.figure()
plt.grid(True)
plt.plot(m_features_list[:3], m_feat_y[:3], 'r-', marker='o', label='Меньшее число признаков')
plt.plot(m_features_list[3:], m_feat_y[3:], 'g-', marker='o', label='Большее число признаков')
plt.title('Зависимость качества от числа признаков')
plt.legend(loc='lower right')
# plt.show()

m_depth_list = [5, 6, 10, 50, 100]
m_depth_y = []
for m_depth in m_depth_list:
    classifer = ensemble.RandomForestClassifier(n_estimators=100, max_depth=m_depth)
    res = model_selection.cross_val_score(classifer, X, y, cv=5)
    m_depth_y.append(res.mean())
classifer = ensemble.RandomForestClassifier(n_estimators=100)
res = model_selection.cross_val_score(classifer, X, y, cv=5)
m_depth_y.append(res.mean())
plt.figure()
plt.grid(True)
plt.plot(m_depth_list[:3], m_depth_y[:3], 'r-', marker='o', label='Меньшая глубина')
plt.plot(m_depth_list[3:5], m_depth_y[3:5], 'y-', marker='o', label='Большая глубина')
plt.plot(m_depth_list[-1], m_depth_y[-1], 'g-', marker='o', label='Неограниченная глубина')
plt.title('Зависимость качества от глубины деревьев')
plt.legend(loc='lower right')
plt.show()

answer_5 = '2 3 4 7'
print('answer_5 = ', answer_5)
write_answer(answer_5, 5)