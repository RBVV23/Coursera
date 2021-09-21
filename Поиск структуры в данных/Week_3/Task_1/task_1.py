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




