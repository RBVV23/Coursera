import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import pairwise_distances


data = pd.read_csv("train.csv", na_values="NaN")
print(data.head())

real_features = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
                 "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]
discrete_features = ["Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32"]
cat_features = data.columns.drop(real_features).drop(discrete_features).drop(["Id", "Response"]).tolist()

print(data[real_features].describe())

complete_real_features = []
less_half_complete_real_features = []
scaled_real_features = []
df = data[real_features].describe()
for column in real_features:
    if df[column][0] == data.shape[0]:
        complete_real_features.append(column)
    elif df[column][0] < 0.5 * data.shape[0]:
        less_half_complete_real_features.append(column)

print('{} ({} шт.)'.format(complete_real_features, len(complete_real_features)))
print('{} ({} шт.)'.format(less_half_complete_real_features, len(less_half_complete_real_features)))

print('data.shape = ', data.shape)

data[real_features].hist(bins=100, figsize=(20,20))
data[discrete_features].hist(bins=100, figsize=(10,10))
plt.show()

seaborn.pairplot(data[real_features+["Response"]].drop(
        ["Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Product_Info_4"], axis=1),
        hue="Response", diag_kind="kde")
plt.show()

seaborn.pairplot(data[discrete_features+['Response']], hue='Response', diag_kind='kde')

seaborn.heatmap(data[real_features].corr(), square=True)

matrix = data[real_features].corr()
for j, feature in enumerate(real_features):
    for i, corr in enumerate(matrix[feature][j+1:]):
        if abs(corr) > 0.9:
            print('Корреляция {} и {} равна (по модулю): {}'.format(feature, real_features[i+j+1], corr))

fig, axes = plt.subplots(11, 10, figsize=(20, 20), sharey=True)
for i in range(len(cat_features)):
    seaborn.countplot(x=cat_features[i], data=data, ax=axes[i // 10, i % 10])
plt.show()

constant_cat_features = []
more_five_cat_features = []

for feature in cat_features:
    if len(np.unique(data[feature])) == 1:
        constant_cat_features.append(feature)
        print('#{} - {}'.format(cat_features.index(feature), feature))
        seaborn.countplot(x=data[feature], data=data)
        plt.show()
    if len(np.unique(data[feature])) > 5:
        more_five_cat_features.append(feature)
        print('#{} - {}'.format(cat_features.index(feature), feature))
        seaborn.countplot(x=data[feature], data=data)
        plt.show()

print(constant_cat_features)
print(more_five_cat_features)

features = ['Medical_Keyword_23', 'Medical_Keyword_39', 'Medical_Keyword_45']

for feature in features:
    seaborn.countplot(x=feature, data=data, hue='Response')
    plt.show()

for feature in features:
    T = data[ (data[feature] == 1) & (data['Response'] == 8) ].shape
    Fs = []
    for lvl in np.unique(data['Response'])[:-1]:
        F = data[ (data[feature] == 1) & (data['Response'] == lvl) ].shape
        Fs.append(F)
    if T >= max(Fs):
        print(feature)

seaborn.countplot(x=data.Response)
plt.show()

sdata = shuffle(data, random_state=321)
del data

subset_l  = 1000
selected_features = real_features[:-4]
objects_with_nan = sdata.index[np.any(np.isnan(sdata[selected_features].values), axis=1)]
data_subset = scale(sdata[selected_features].drop(objects_with_nan, axis=0)[:subset_l])
response_subset = sdata["Response"].drop(objects_with_nan, axis=0)[:subset_l]

tsne = TSNE(random_state=321)
tsne_representation = tsne.fit_transform(data_subset)

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(sdata.Response), colors):
    plt.scatter(tsne_representation[response_subset.values==y, 0],
                tsne_representation[response_subset.values==y, 1], color=c, alpha=0.5, label=str(y))
plt.legend(loc='lower left')
plt.show()

mds = MDS(random_state=321)
MDS_transformed = mds.fit_transform(data_subset)

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(response_subset), colors):
    plt.scatter(MDS_transformed[response_subset.values==y, 0],
                MDS_transformed[response_subset.values==y, 1],
                color=c, alpha=0.5, label=str(y))
plt.legend()
plt.show()

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(response_subset), colors):
    plt.scatter(MDS_transformed[response_subset.values==y, 0],
                MDS_transformed[response_subset.values==y, 1],
                color=c, alpha=0.5, label=str(y))
plt.legend()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()


matrix = pairwise_distances(data_subset, metric='cosine')
mds = MDS(random_state=321, dissimilarity="precomputed")
MDS_transformed_cos = mds.fit_transform(matrix)

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(response_subset), colors):
    plt.scatter(MDS_transformed_cos[response_subset.values[:subset_l]==y, 0],
                MDS_transformed_cos[response_subset.values[:subset_l]==y, 1],
                color=c, alpha=0.5, label=str(y))
plt.legend()
plt.show()

person_features = ["Ins_Age", "Ht", "Wt", "BMI"]
svm_ = svm.OneClassSVM(gamma=10, nu=0.01)
svm_.fit(sdata[person_features])
labels = svm_.predict(sdata[person_features])
print((labels==1).mean())

plt.subplots(2, 3, figsize=(12, 8))
n = 0
for i in range(len(person_features)-1):
    for j in range(i+1,len(person_features)):
        n += 1
        plt.subplot(2,3,n)
        plt.scatter(sdata[person_features[i]][labels == 1], sdata[person_features[j]][labels == 1], c='blue', alpha=0.5)
        plt.scatter(sdata[person_features[i]][labels == -1], sdata[person_features[j]][labels == -1], c='red', alpha=0.5)
        plt.xlabel(person_features[i])
        plt.ylabel(person_features[j])
plt.show()

features = ['BMI', 'Employment_Info_1', 'Medical_History_32']
for feature in features:
    seaborn.distplot(sdata[feature], bins=50) # функция исчезнет при обновлении библиотеки (текущая версия: 0.11.1)
    plt.show()

