import pandas as pd
import numpy as np

def my_write_answer(answer, part, number):
    name = 'answer' + str(part) + str(number) + '.txt'
    with open(name, 'w') as file:
        file.write(str(answer))

pre_data = pd.read_csv('coursera_sessions_train.txt', header=None, engine='python', sep='/n')

with open('coursera_sessions_train.txt') as f:
    all_sessions = f.read().split('\n')

res_sessions = all_sessions
print(all_sessions)

for i, session in enumerate(reversed(all_sessions)):
    if session[-1] == ';':
        res_sessions.pop(i)

print(res_sessions)



# print(train_data)
# print(train_data[:10])
