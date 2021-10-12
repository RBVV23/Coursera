import numpy as np

np.random.seed(1)

statistical_population = np.random.randint(2, size=10000)
random_sample = np.random.choice(statistical_population, size=1000)

print('Истинное значение доли единиц в популяции: ', statistical_population.mean())