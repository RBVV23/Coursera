import numpy as np
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import samplesize_confint_proportion

np.random.seed(1)
statistical_population = np.random.randint(2, size=10000)

random_sample = np.random.choice(statistical_population, size=1000)

print('Истинное значение доли единиц в популяции: ', statistical_population.mean())
print('Точечная оценка доли единиц (по выборке): ', random_sample.mean())

normal_interval = proportion_confint(count=sum(random_sample), nobs=len(random_sample),
                                     method='normal')
print('Нормальный интервал [{}; {}] с шириной {}'.format(normal_interval[0],
                                                         normal_interval[1],
                                                         normal_interval[1]-normal_interval[0]))

wilson_interval = proportion_confint(count=sum(random_sample), nobs=len(random_sample),
                                     method='wilson')
print('Вильсоновский интервал [{}; {}] с шириной {}'.format(wilson_interval[0], wilson_interval[1], wilson_interval[1]-wilson_interval[0]))

n_samples = int(np.ceil(samplesize_confint_proportion(proportion=random_sample.mean(),
                                                      half_length=0.01)))
