import numpy as np
# print(np.__version__)
from scipy import spatial

import re

with open('sentences.txt') as file:
    # print(file.read())
    # all_text = file
    text = file.readlines()

words = []
text_table = []
print(type(text_table))
for string in text:
    string = string.lower()
    line = re.split('[^a-z]', string)
    text_table.append(line)
    words += line

# print(text_table[0][1])

while True:
    try:
        words.remove('')
    except ValueError:
        break
# print(words)
print(len(words))

d = 0
my_dict = { }
# my_dict = {0: 'in', 2:'comparison'}

for word in words:
    if (word not in my_dict.values()):
        my_dict[d] = word
        d += 1

# print(my_dict)

matrix = np.zeros((len(text), d))
print(matrix.shape)
# print(matrix)

i = 0 # количество предложений (строк матрицы)
j = 0 # количество слов в словаре (столбцов матрицы)
for line in text_table:
    # print(line)
    for j in range(d):
        matrix[i][j] = line.count(my_dict[j])
    i += 1

# print(matrix[0])
# print(text[2])
# print(my_dict[2])
dist = []
for i in range(len(text)):
    ans = spatial.distance.cosine(matrix[0], matrix[i])
    # print(ans)
    dist.append(ans)


for i in range(4):
    print(str(i+1) + '. Расстояние до предложения №' + str(dist.index(min(dist))) + ' составляет ' + str(min(dist)))
    dist[dist.index(min(dist))] += 1

