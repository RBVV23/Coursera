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
print('Вильсоновский интервал [{}; {}] с шириной {}'.format(wilson_interval[0],
                                                            wilson_interval[1],
                                                            wilson_interval[1]-wilson_interval[0]))

n_samples = int(np.ceil(samplesize_confint_proportion(proportion=random_sample.mean(),
                                                      half_length=0.01)))
print('Объём выборки необходимый для построения интервала шириной 0.02:', n_samples)

np.random.seed(1)
new_random_sample = np.random.choice(statistical_population, size=n_samples)
new_normal_interval = proportion_confint(count=sum(new_random_sample), nobs=len(new_random_sample),
                                     method='normal')
print('Нормальный интервал [{}; {}] (с наперёд заданной) шириной {}'.format(new_normal_interval[0],
                                                         new_normal_interval[1],
                                                         new_normal_interval[1]-new_normal_interval[0]))


