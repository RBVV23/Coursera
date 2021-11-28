import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


fin = open('fertility.txt', 'r')
data = list(map(lambda x: int(x.strip()),fin.readlines()))

print('data[:20]: ')
print(data[:20])

# print(range(max(data)))
# print(max(data))
plt.bar(range((max(data))+1), np.bincount(data), color='b', label='Количество детей')
plt.legend()
plt.show()

l = np.mean(data)
print('Среднее значение: ', l)

observed_frequences = np.bincount(data)
print('Наблюдаемые частоты: ')
print((observed_frequences))

expected_frequences = [len(data)*stats.poisson.pmf(x,l) for x in range(min(data), max(data)+1)]
print('Ожидаемые частоты: ')
print((expected_frequences))

plt.bar(range(len(expected_frequences)), expected_frequences, color='b', label='Распределение Пуассона')
plt.legend()
plt.show()

print('Вероятность получения данной (или более экстремальной) статистики при условии нулевой гипотезы (величина подчиняется распределению Пуассона):')
result = stats.chisquare(observed_frequences, expected_frequences, ddof=1)
print(result)
p_value = result[1]
print('p-value = ', p_value*100, '%')