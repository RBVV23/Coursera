import numpy as np
from sklearn import model_selection, metrics, datasets, linear_model, tree, ensemble
from sklearn.model_selection import train_test_split

def write_answer(answer, number):
    name = 'ans{}.txt'.format(number)
    with open(name, "w") as fout:
        fout.write(str(answer))

def my_euclid_dist(obj_1, obj_2):
    if len(obj_1) == len(obj_2):
        L = len(obj_1)
    else:
        raise IndexError('Объекты имеют разную размерность')
        return
    dist = 0
    for i in range(L):
        dist += (obj_1[i] - obj_2[i])**2
    dist = dist**0.5
    return dist

def my_1NN(obj, X, y):
    N = X.shape[0]
    matrix = np.ones((N,2))*-1
    for i in range(N):
        matrix[i][0] = my_euclid_dist(obj, X[i])
        matrix[i][1] = y[i]
    ind = matrix[:,0].argmin()
    return int(matrix[ind,1])

def my_error_counter(y_true, y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    return np.sum([y_true != y_predict]) / len(y_true)




digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

y_predict = []
for obj in X_test:
    prediction = my_1NN(obj, X_train, y_train)
    y_predict.append(prediction)


# print('y_test: ', y_test)
# print('y_predict: ', y_predict)
answer_1 = my_error_counter(y_test, y_predict)
print('answer_1 = ', answer_1)
write_answer(answer_1, 1)


classifer = ensemble.RandomForestClassifier(n_estimators=1000, random_state=0)
classifer.fit(X_train, y_train)
y_predict = classifer.predict(X_test)
answer_2 = my_error_counter(y_test, y_predict)
print('answer_2 = ', answer_2)
write_answer(answer_2, 2)
