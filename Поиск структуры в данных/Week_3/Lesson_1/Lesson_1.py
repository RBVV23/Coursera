from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import random_projection
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from collections import Counter


digits = datasets.load_digits()

print(digits.DESCR)

print('target: ', digits.target[0])
print('data:')
print(digits.data[0])
print('number of features: ', len(digits.data[0]))

print(digits.data[0].shape)
print(digits.data[0].reshape(8,8))
print(digits.data[0].reshape(8,8).shape)

plt.imshow(digits.data[0].reshape(8,8))
plt.show()

print('digits.keys(): ')
print(digits.keys())
plt.imshow(digits.images[0].reshape(8,8))
plt.show()


plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.imshow(digits.images[0])
plt.subplot(2,2,2)
plt.imshow(digits.images[0], cmap='hot')
plt.subplot(2,2,3)
plt.imshow(digits.images[0], cmap='gray')
plt.subplot(2,2,4)
plt.imshow(digits.images[0], cmap='gray', interpolation='nearest')
plt.show()

plt.figure(figsize=(20,8))

for plot_number, plot in enumerate(digits.images[:10]):
    plt.subplot(2,5, plot_number+1)
    plt.imshow(plot, cmap='gray')
    plt.title('digit: ' + str(digits.target[plot_number]))
plt.show()

data = digits.data[:1000]
labels = digits.target[:1000]

print(Counter(labels))
# print(Counter(labels).keys())
# print(Counter(labels).values())

plt.figure(figsize=(10,6))
plt.bar(Counter(labels).keys(), Counter(labels).values())
plt.show()

classifer = KNeighborsClassifier()
print(classifer)
classifer.fit(data, labels)
print(classification_report(classifer.predict(data), labels))
# print(classification_report(classifer.predict(digits.data[1000:2000]), digits.target[1000:2000]))

projection = random_projection.SparseRandomProjection(n_components=2, random_state=0)
data_2d_rp = projection.fit_transform(data)

