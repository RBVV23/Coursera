import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn

data = pandas.read_csv("train.csv", na_values="NaN")
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

# Код 1. Постройте гистограммы.
data[real_features].hist(bins=100, figsize=(20,20))
data[discrete_features].hist(bins=100, figsize=(10,10))
plt.show()

seaborn.pairplot(data[real_features+["Response"]].drop(
        ["Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Product_Info_4"], axis=1),
        hue="Response", diag_kind="kde")
plt.show()

# Код 2. Постройте pairplot для целочисленных признаков
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

# Код 3. Постройте countplot
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



