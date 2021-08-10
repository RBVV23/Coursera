# PYTHON LVL 0


import codecs

file = codecs.open('to_read.txt', 'r', encoding='utf-8')
print(file.read())
file.close()

file = codecs.open('to_read.txt', 'r', encoding='utf-8')
for line in file:
    print(line)
file.close()

file = codecs.open('to_read.txt', 'r', encoding='utf-8')
list_1 = []
list_2 = []
list_1 = file.readlines()
file.close()

print(list_1)
for line in list_1:
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


list_3 = [x**2 for x in range(100) if (x % 2 == 0) and (x % 3 == 0)]
print(list_3)
list_4 = list(map(lambda x: int(x**(0.5)), list_3))
print(list_4)
