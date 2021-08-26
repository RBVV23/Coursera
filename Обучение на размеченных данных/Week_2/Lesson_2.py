import numpy as np
from sklearn import model_selection, datasets


iris = datasets.load_iris()

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(iris.data,
                                                                iris.target, test_size=0.3)
print(len(test_labels)/len(iris.data))
print('Размер обучающей выборки: {} объектов'.format(len(train_data)))
print('Размер тестовой выборки: {} объектов'.format(len(test_data)))

print('Обучающая выборка: \n', train_data[:5], '\n')
print('Тестовая выборка: \n', test_data[:5], '\n')

print('Метки классов на обучающей выборке: \n', train_labels, '\n')
print('Метки классов на тестовой выборке: \n', test_labels, '\n')


X = range(10)
kf = model_selection.KFold(n_splits=5)
print('\nKFold:')
for train_indices, test_indices in kf.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

kf = model_selection.KFold(n_splits=2, shuffle=True)
print('\nKFold (shuffle=True):')
for train_indices, test_indices in kf.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

kf = model_selection.KFold(n_splits=2, shuffle=True, random_state=1)
print('\nKFold (random_state=1):')
for train_indices, test_indices in kf.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

target = np.array([0]*5 + [1]*5)
print('\n', target)

skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
print('\nStratifiedKFold:')
for train_indices, test_indices in skf.split(X, target):

    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

ss = model_selection.ShuffleSplit(n_splits = 10, test_size=0.2)
print('\nShuffleSplit:')
for train_indices, test_indices in ss.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

sss = model_selection.StratifiedShuffleSplit(n_splits = 4, test_size=0.2)
print('\nStratifiedShuffleSplit:')
for train_indices, test_indices in sss.split(X, target):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))

loo = model_selection.LeaveOneOut()
print('\nLeaveOneOut:')
for train_indices, test_indices in loo.split(X):
    print('Обучение: {} \t Тест: {}'.format(train_indices, test_indices))
