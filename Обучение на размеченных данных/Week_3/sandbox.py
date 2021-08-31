import numpy as np
import math as m

def bernulli(k, n, p):
    nk = m.factorial(n)/(m.factorial(n-k)*m.factorial(k))
    q = 1 - p
    return nk*(p**k)*(q**(n-k))

p = 0.2
n = 15
k = 4
a = bernulli(k, n, p)**4
print(bernulli(k, n, p))
p = 0.8
n = 1
k = 1
b = bernulli(k, n, p)**11
print(bernulli(k, n, p)**11)

print(a+b)


