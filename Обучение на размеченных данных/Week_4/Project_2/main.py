import numpy as np
from sklearn import model_selection, metrics, datasets, linear_model, tree
import xgboost as xgb

def write_answer(answer, number):
    name = 'ans{}.txt'.format(number)
    with open(name, "w") as fout:
        fout.write(str(answer))

def my_MSE_error(y, y_predict):
    y = np.array(y)
    y_predict = np.array(y_predict)
    MSE = sum((1/len(y))*(y - y_predict)**2)
    return MSE

def gbm_predict(X):
    return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list, coefficients_list)]) for x in X]

boston = datasets.load_boston()

data = boston.data
print('data.shape = ', data.shape)
target = boston.target
print('target.shape = ', target.shape)
print(boston.feature_names)

X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, shuffle=False, test_size = 0.25)

y = y_train
y_predict = y_train.mean()
my_MSE_error(y, y_predict)

error_vec = y - y_predict
grad_vec = -2*(y_predict - y)

base_algorithms_list = []
coefficients_list= []
y_new = error_vec
y = y_train

for i in range(50):
    estimator = tree.DecisionTreeRegressor(max_depth=5, random_state=42)
    estimator.fit(X_train, y)
    base_algorithms_list.append(estimator)
    coefficients_list.append(0.9)
    y_predict = gbm_predict(X_train)
#     print('y_predict:')
#     print(y_predict)
    y_new = -2*(y_predict - y_train)
    y = y_new
#     print('gbm_predict:')
#     print(gbm_predict(X_train))
#     print('iteration #', i)
#     print('current error: ', metrics.mean_squared_error(gbm_predict(X_train), y_train))


answer_2 = metrics.mean_squared_error(gbm_predict(X_test), y_test)**0.5
print('answer_2 = ',answer_2)
write_answer(answer_2, 2)

base_algorithms_list = []
coefficients_list= []
y_new = error_vec
y = y_train

for i in range(50):
    estimator = tree.DecisionTreeRegressor(max_depth=5, random_state=42)
    estimator.fit(X_train, y)
    base_algorithms_list.append(estimator)
    new_coeff = 0.9/(1+i)
    coefficients_list.append(new_coeff)
    y_predict = gbm_predict(X_train)
    y_new = -2*(y_predict - y_train)
    y = y_new
#     print('iteration #', i)
#     print('current error: ', metrics.mean_squared_error(gbm_predict(X_train), y_train))


answer_3 = metrics.mean_squared_error(gbm_predict(X_test), y_test)**0.5
print('answer_3 = ',answer_3)
write_answer(answer_3, 3)

n_trees = [10, 20, 30] + list(range(50,1001,50))

n_tree_errors = []
for n_tree in n_trees:
    estimator = xgb.XGBRegressor(n_estimators=n_tree, max_depth=5)
    estimator.fit(X_train, y_train)
    n_tree_errors.append(metrics.mean_squared_error(estimator.predict(X_test), y_test))
#     print(n_tree)
print("n_tree_errors: ")
print(n_tree_errors)

depths = [1, 2, 3] + list(range(5,301,15))

depth_errors = []
for depth in depths:
    estimator = xgb.XGBRegressor(n_estimators=50, max_depth=depth)
    estimator.fit(X_train, y_train)
    depth_errors.append(metrics.mean_squared_error(estimator.predict(X_test), y_test))
#     print(depth)
print("depth_errors: ")
print(depth_errors)

answer_4 = '2 3'
write_answer(answer_4, 4)
print('answer_4 = ', answer_4)

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

answer_5 = metrics.mean_squared_error(regressor.predict(X_test), y_test)
answer_5 = answer_5**0.5
write_answer(answer_5, 5)
print('answer_5 = ', answer_5)