import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score as cv_score
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces

def my_corr_pirson(v1, v2, debug=False):
    np.array(v1)
    np.array(v2)
    mn1 = np.mean(v1)
    mn2 = np.mean(v2)
    if debug:
        print('np.mean(v1) = ', np.mean(v1))
        print('np.mean(v2) = ', np.mean(v2))
    ch = 0
    zn1 = 0
    zn2 = 0
    for i in range(len(v1)):
        (v1[i] - mn1) * ((v2[i] - mn2))
        if debug:
                print('v1[{}]-mn1 = {} - {} = {}'.format(i, v1[i],mn1, (v1[i]-mn1)))
                print('v2[{}]-mn2 = {} - {} = {}'.format(i, v2[i],mn2, (v2[i]-mn2)))
                print('\t(v1[{}]-mn1)*((v2[{}]-mn2)) = {}'.format(i, i, (v1[i]-mn1)*((v2[i]-mn2))))
        ch += (v1[i] - mn1) * (v2[i] - mn2)
        zn1 += (v1[i] - mn1) ** 2
        zn2 += (v2[i] - mn2) ** 2
        if debug:
            print('(v1[{}]-mn1)**2 = {} - {} = {}'.format(i, v1[i],mn1, (v1[i]-mn1)**2))
            print('(v2[{}]-mn2)**2 = {} - {} = {}'.format(i, v2[i],mn2, (v2[i]-mn2)**2))
    zn = (zn1 * zn2) ** 0.5
    if debug:
        print('ch = ', ch)
        print('zn1 = ', zn1)
        print('zn2 = ', zn2)
        print('zn = ', zn)
    return ch / zn

def plot_principal_components(data, model, scatter=True, legend=True):
    W_pca = model.components_
    if scatter:
        plt.scatter(data[:,0], data[:,1])
    plt.plot(data[:,0], -(W_pca[0,0]/W_pca[0,1])*data[:,0], color="c")
    plt.plot(data[:,0], -(W_pca[1,0]/W_pca[1,1])*data[:,0], color="c")
    if legend:
        c_patch = mpatches.Patch(color='c', label='Principal components')
        plt.legend(handles=[c_patch], loc='lower right')
    plt.axis('equal')
    limits = [np.minimum(np.amin(data[:,0]), np.amin(data[:,1]))-0.5,
              np.maximum(np.amax(data[:,0]), np.amax(data[:,1]))+0.5]
    plt.xlim(limits[0],limits[1])
    plt.ylim(limits[0],limits[1])
    plt.draw()
def plot_scores(d_scores):
    n_components = np.arange(1, d_scores.size + 1)
    plt.plot(n_components, d_scores, 'b', label='PCA scores')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('cv scores')
    plt.legend(loc='lower right')
    plt.show()
def plot_variances(d_variances):
    n_components = np.arange(1, d_variances.size + 1)
    plt.plot(n_components, d_variances, 'b', label='Component variances')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('variance')
    plt.legend(loc='upper right')
    plt.show()
def plot_iris(transformed_data, target, target_names):
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(transformed_data[target == i, 0],
                    transformed_data[target == i, 1], c=c, label=target_name)
    plt.legend()
    plt.show()

def write_answer_1(optimal_d):
    with open("pca_answer1.txt", "w") as fout:
        fout.write(str(optimal_d))
def write_answer_2(optimal_d):
    with open("pca_answer2.txt", "w") as fout:
        fout.write(str(optimal_d))
def write_answer_3(list_pc1, list_pc2):
    with open("pca_answer3.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc1]))
        fout.write(" ")
        fout.write(" ".join([str(num) for num in list_pc2]))
def write_answer_4(list_pc):
    with open("pca_answer4.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc]))

mu = np.zeros(2)
C = np.array([[3,1],[1,2]])

data = np.random.multivariate_normal(mu, C, size=50)
plt.scatter(data[:,0], data[:,1])
plt.show()

v, W_true = np.linalg.eig(C)

plt.scatter(data[:,0], data[:,1])
plt.plot(data[:,0], (W_true[0,0]/W_true[0,1])*data[:,0], color="g")
plt.plot(data[:,0], (W_true[1,0]/W_true[1,1])*data[:,0], color="g")
g_patch = mpatches.Patch(color='g', label='True components')
plt.legend(handles=[g_patch])
plt.axis('equal')
limits = [np.minimum(np.amin(data[:,0]), np.amin(data[:,1])),
          np.maximum(np.amax(data[:,0]), np.amax(data[:,1]))]
plt.xlim(limits[0],limits[1])
plt.ylim(limits[0],limits[1])
plt.draw()

model = PCA(n_components=2)
model.fit(data)

plt.scatter(data[:,0], data[:,1])

plt.plot(data[:,0], (W_true[0,0]/W_true[0,1])*data[:,0], color="g")
plt.plot(data[:,0], (W_true[1,0]/W_true[1,1])*data[:,0], color="g")

plot_principal_components(data, model, scatter=False, legend=False)
c_patch = mpatches.Patch(color='c', label='Principal components')
plt.legend(handles=[g_patch, c_patch])
plt.draw()

data_large = np.random.multivariate_normal(mu, C, size=5000)

model = PCA(n_components=2)
model.fit(data_large)
plt.scatter(data_large[:,0], data_large[:,1], alpha=0.1)

plt.plot(data_large[:,0], (W_true[0,0]/W_true[0,1])*data_large[:,0], color="g")
plt.plot(data_large[:,0], (W_true[1,0]/W_true[1,1])*data_large[:,0], color="g")

plot_principal_components(data_large, model, scatter=False, legend=False)
c_patch = mpatches.Patch(color='c', label='Principal components')
plt.legend(handles=[g_patch, c_patch])
plt.draw()


data = pd.read_csv('data_task1.csv')
D = data.shape[1]
d_scores = []
for d in range(1, D + 1):
    model = PCA(n_components=d)
    scores = cv_score(model, data)
    #     print(scores)
    #     print(scores.mean())
    d_scores.append(scores.mean())

d_scores = np.array(d_scores)
# print(d_scores.size)
plot_scores(d_scores)

optimal_d = 20
print('answer_1 = ', optimal_d)
write_answer_1(optimal_d)

data = pd.read_csv('data_task2.csv')

D = data.shape[1]
# print('data: ')
# print(data)

model = PCA(n_components=D)
model.fit(data)
transform_data = model.transform(data)
# print('transform_data: ')
# print(transform_data)

d_variances = []


for i in range(D):
#     print('column: ', column, i)
    column = transform_data[:,i]
    column_mean = column.mean()
#     print('column_mean = ',column_mean)
    sum = 0
    for cell in column:
        sum += (cell - column_mean)**2
    variance = sum/len(column)
    d_variances.append(variance)
d_variances = np.array(d_variances)
sort_variances = np.sort(d_variances)[::-1]
# print(sort_variances)
var_diff = []
for i in range(len(sort_variances)-1):
    diff = sort_variances[i] - sort_variances[i+1]
    var_diff.append(diff)
print(var_diff)
plot_variances(sort_variances)
print(max(var_diff))
print(var_diff.index(max(var_diff)))
write_answer_2(var_diff.index(max(var_diff))+1)
print('answer_2 = ', var_diff.index(max(var_diff))+1)


iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

centered_data = data - np.mean(data, axis=0)
# print('centered_data: ')
# print(centered_data)

model = PCA(n_components=2)
model.fit(data)
transformed_data = model.transform(data)

plot_iris(transformed_data, target, target_names)
centered_target = target - target.mean()
# print('centered_target:')
# print(centered_target)
centered_transformed_data = transformed_data - np.mean(transformed_data, axis=0)


with_1 = []
with_2 = []

for i in range(centered_data.shape[1]):
    column = centered_data[:, i]
    # print('Корреляция с 1-ым компонентом: ', my_corr_pirson(column, centered_transformed_data[:, 0]))
    with_1.append(my_corr_pirson(column, centered_transformed_data[:, 0]))
    # print('Корреляция со 2-ым компонентом: ', my_corr_pirson(column, centered_transformed_data[:, 1]))
    with_2.append(my_corr_pirson(column, centered_transformed_data[:, 1]))

print('with_1: ', with_1)
print('with_2: ', with_2)
list_pc1 = []
list_pc2 = []

for i in range(len(with_1)):
    if abs(with_1[i]) > abs(with_2[i]):
        list_pc1.append(i + 1)
    else:
        list_pc2.append(i + 1)

print(list_pc1)
print(list_pc2)

write_answer_3(list_pc1, list_pc2)
answer_3 = list_pc1.append(list_pc2)
print('answer_3 = ', answer_3)


data = fetch_olivetti_faces(shuffle=True, random_state=0).data
image_shape = (64, 64)

model = PCA(n_components=10, svd_solver='randomized')
model.fit(data)
transformed_data = model.transform(data)
centered_transformed_data = transformed_data - np.mean(transformed_data, axis=0)

centered_data = data - np.mean(data, axis=0)
# print('centered_data.shape = ',centered_data.shape)

components = centered_transformed_data

components = components ** 2
vec = components.sum(axis=1)
# print('len(vec) = ', len(vec))


list_pc = []
for i, var in enumerate(vec):
    components[i, :] = components[i, :] / var
#     print(np.argmax(components[k,:]))

for k in range(components.shape[1]):
    list_pc.append(np.argmax(components[:, k]))
    print(np.argmax(components[:, k]))

write_answer_4(list_pc)
answer_4 = list_pc
print('answer_4 = ', answer_4)

for id in list_pc:
    image = data[id]
    plt.figure()
    plt.imshow(image.reshape(image_shape))


C1 = np.array([[10,0],[0,0.5]])
phi = np.pi/3
C2 = np.dot(C1, np.array([[np.cos(phi), np.sin(phi)],
                          [-np.sin(phi),np.cos(phi)]]))

data = np.vstack([np.random.multivariate_normal(mu, C1, size=50),
                  np.random.multivariate_normal(mu, C2, size=50)])
plt.scatter(data[:,0], data[:,1])
plt.plot(data[:,0], np.zeros(data[:,0].size), color="g")
plt.plot(data[:,0], 3**0.5*data[:,0], color="g")
model = PCA(n_components=2)
model.fit(data)
plot_principal_components(data, model, scatter=False, legend=False)
c_patch = mpatches.Patch(color='c', label='Principal components')
plt.legend(handles=[g_patch, c_patch])
plt.draw()

C = np.array([[0.5,0],[0,10]])
mu1 = np.array([-2,0])
mu2 = np.array([2,0])

data = np.vstack([np.random.multivariate_normal(mu1, C, size=50),
                  np.random.multivariate_normal(mu2, C, size=50)])
plt.scatter(data[:,0], data[:,1])
model = PCA(n_components=2)
model.fit(data)
plot_principal_components(data, model)
plt.draw()