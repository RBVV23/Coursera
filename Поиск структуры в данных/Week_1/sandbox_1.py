import numpy as np
import itertools

permutations = itertools.permutations([0, 1, 2])
# print(list(permutations))

for a,b,c in permutations:
    print('a = {}, b = {}, c = {}'.format(a,b,c))
    mapping = mapping = {2:a, 1:b, 0:c}
    print(mapping)