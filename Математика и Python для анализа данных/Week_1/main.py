# PYTHON LVL 0
# Чтение данных из файлов

import codecs

file = codecs.open('to_read.txt', 'r', encoding='utf-8')
print(file.read())
file.close()

file = codecs.open('to_read.txt', 'r', encoding='utf-8')
for line in file:
    print(line)
file.close()

file = codecs.open('to_read.txt', 'r', encoding='utf-8')
list = []
list_2 = []
list = file.readlines()
file.close()

print(list)
for line in list:
    print(line.strip())
    list_2.append(line.strip())

list_2 = list_2[::-1]
print(list_2)

file_2 = open('to_write.txt', 'w')
file_2.write('В обратном порядке:\n')
file_2.close()

file_2 = open('to_write.txt', 'a')
for line in list_2:
    file_2.write(line + '\n')
file_2.close()


list_3 = [x**2 for x in range(101) if (x % 2 == 0) and (x % 3 == 0)]
# print(list_3)
numbers = [1, 3, 5]
list_4 = [ map(lambda x: x**3, numbers) ]
print(type(list_4))
print(list_4)