import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np

import matplotlib.pyplot as plt

from pybrain.datasets import ClassificationDataSet # Структура данных pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError
#для корректной работы требуется магия со старой версией библиотеки

def plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec):
    plt.figure()
    plt.plot(hidden_neurons_num, res_train_vec)
    plt.plot(hidden_neurons_num, res_test_vec, '-r')
def write_answer_nn(optimal_neurons_num):
    with open("nnets_answer1.txt", "w") as fout:
        fout.write(str(optimal_neurons_num))

with open('winequality-red.csv') as f:
    f.readline()  # пропуск заголовочной строки
    data = np.loadtxt(f, delimiter=';')

TRAIN_SIZE = 0.7 # Разделение данных на обучающую и контрольную части в пропорции 70/30%
from sklearn.model_selection import train_test_split

y = data[:, -1]
np.place(y, y < 5, 5)
np.place(y, y > 7, 7)
y -= min(y)
X = data[:, :-1]
X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0)

HIDDEN_NEURONS_NUM = 100
MAX_EPOCHS = 100

ds_train = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))

ds_train.setField('input', X_train)
ds_train.setField('target', y_train[:, np.newaxis])
ds_train._convertToOneOfMany( )
ds_test = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
ds_test.setField('input', X_test)
ds_test.setField('target', y_test[:, np.newaxis])
ds_test._convertToOneOfMany( )

np.random.seed(0)
net = buildNetwork(ds_train.indim, HIDDEN_NEURONS_NUM, ds_train.outdim, outclass=SoftmaxLayer)


random.seed(0)

trainer = BackpropTrainer(net, dataset=ds_train)

err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)

plt.plot(err_train, 'b', label='train')
plt.plot(err_val, 'r', label='validation')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.legend()
plt.show()

res_train = net.activateOnDataset(ds_train).argmax(axis=1)
print ('Error on train: ', percentError(res_train, ds_train['target'].argmax(axis=1)), '%')
res_test = net.activateOnDataset(ds_test).argmax(axis=1)
print ('Error on test: ', percentError(res_test, ds_test['target'].argmax(axis=1)), '%')

random.seed(0)
np.random.seed(0)

hidden_neurons_num = [50, 100, 200, 500, 700, 1000]
res_train_vec = list()
res_test_vec = list()

MAX_EPOCHS = 100

for nnum in hidden_neurons_num:
    HIDDEN_NEURONS_NUM = nnum
    random.seed(0)
    np.random.seed(0)
    net = buildNetwork(ds_train.indim, HIDDEN_NEURONS_NUM, ds_train.outdim, outclass=SoftmaxLayer)
    init_params = np.random.random((len(net.params)))
    net._setParameters(init_params)
    trainer = BackpropTrainer(net, dataset=ds_train)
    err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
    res_train = net.activateOnDataset(ds_train).argmax(axis=1)
    res_train_vec.append(percentError(res_train, ds_train['target'].argmax(axis=1)))
    res_test = net.activateOnDataset(ds_test).argmax(axis=1)
    res_test_vec.append(percentError(res_test, ds_test['target'].argmax(axis=1)))
    print(nnum)


plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec)
write_answer_nn(hidden_neurons_num[res_test_vec.index(min(res_test_vec))])
print(hidden_neurons_num[res_test_vec.index(min(res_test_vec))])