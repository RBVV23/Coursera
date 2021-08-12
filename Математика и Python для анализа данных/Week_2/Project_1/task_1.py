import numpy as np
print(np.__version__)
import scipy as sp
print(sp.__version__)
import re

with open('sentences.txt') as file:
    # print(file.read())
    # all_text = file
    text = file.readlines()

words = []
for string in text:
    string = string.lower()
    line = re.split('[^a-z]', string)
    words += line

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

matrix = np.array()

