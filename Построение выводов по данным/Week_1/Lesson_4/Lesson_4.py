import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def my_get_boostraps_samples(data, n_samples):
    L = len(data)
    indices = np.random.randint(0, L, (n_samples, L))
    samples = data[indices]
    return samples
def my_stat_intervals(stat, alpha):
    low, high = np.percentile(stat, [100*alpha/2., 100*(1 - alpha/2.)])
    return low, high

data = pd.read_csv('verizon.txt', sep='\t')
print('data.shape = ', data.shape)
print(data.head())

print(data.Group.value_counts())

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(data[data['Group'] == 'ILEC'].Time, bins=20, color='b', range=(0,100), label='ILEC')
plt.legend()

plt.subplot(1,2,2)
plt.hist(data[data['Group'] == 'CLEC'].Time, bins=20, color='r', range=(0,100), label='CLEC')
plt.legend()
# plt.show()

# indices = np.random.randint(0, 10, (3, 10))
# print(indices)
#
# arr = np.array([0,1,2,3,4,5,6,7,8,9])
# ind = np.array([[1,2,3],[3,4,5],[4,5,6]])
# print(arr[ind])

ilec_time = data[data.Group == 'ILEC'].Time.values
clec_time = data[data.Group == 'CLEC'].Time.values


# print(my_stat_intervals([1,2,3,4], 0.05))
np.random.seed(0)

ilec_median_score = list(map(np.median, my_get_boostraps_samples(ilec_time, 1000)))
print('95% доверительный интервал "ilec_median_score": ', my_stat_intervals(ilec_median_score,0.05))
clec_median_score = list(map(np.median, my_get_boostraps_samples(clec_time, 1000)))
print('95% доверительный интервал "clec_median_score": ', my_stat_intervals(clec_median_score,0.05))

print('Точечная оценка разности медиан: ', np.median(clec_time) - np.median(ilec_time))

delta_median_score = list(map(lambda x: x[0] - x[1], zip(clec_median_score, ilec_median_score)))
print('Интервальная оценка разности медиан: ', my_stat_intervals(delta_median_score,0.05))