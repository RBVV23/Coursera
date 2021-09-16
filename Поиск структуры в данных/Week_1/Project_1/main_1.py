import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

def write_answer(answer):
    name = 'answer.txt'
    with open(name, "w") as fout:
        fout.write(str(answer))

def dist_to_nearest_office(offices, cluster_center, debug=False):
    if debug:
        print('Center: X = {}; Y = {}'.format(cluster_center[0], cluster_center[1]))
    dists = []
    for item in offices:
        dist = ((item[0] - cluster_center[0])**2 + (item[1] - cluster_center[1])**2 )**0.5
        dists.append(dist)
        if debug:
            print('\tOffice: X = {}; Y = {}'.format(item[0], item[1]))
            print('\tDistantion = ', dist)
    min_dist = min(dists)
    ind = dists.index(min_dist)
    if debug:
        print('\t\tNearest office #{}: {}'.format(ind, offices[ind]))
        print('\t\tMinimal distance: {}'.format(min_dist))
    return min_dist, ind

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 100)

debug=True

data = pd.read_csv('checkins.csv', sep='|', header=0, skipinitialspace=True)
columns = data.columns
# print('data.shape = ', data.shape)
data = data.drop(data.index[[0, -1]])

coordinates_of_offices = [[33.751277, -118.188740],
                          [25.867736, -80.324116],
                          [51.503016, -0.075479],
                          [52.378894, 4.885084],
                          [39.366487, 117.036146],
                          [-33.868457, 151.205134]]
cities_offices = ['Los Angeles', 'Miami', 'London', 'Amsterdam', 'Beijing', 'Sydney']

my_columns = []
for col in columns:
    new_name = col.strip()
    my_columns.append(new_name)
    data = data.rename(columns={col: new_name})

data = data.astype({'id': int, 'user_id': int, 'venue_id': int})

print(data.head())
data = data.dropna(subset=['latitude', 'longitude'])

# print(data.head())
print('data.shape = ', data.shape)

small_data = data.iloc[0:100000]
data = data.iloc[:30000]
# data = data.iloc[0:100000]

# print(small_data.head())
# print('small_data.shape = ', small_data.shape)

X = data.iloc[:, [3,4]]
# print(X.head())
clustering = MeanShift(bandwidth=0.1).fit(X)
# print('clustering.labels: ', clustering.labels_)
# print('clustering.cluster_centers: ', clustering.cluster_centers_[:10])
cluster_centers = clustering.cluster_centers_
unique, counts = np.unique(clustering.labels_, return_counts=True)
print('Количество кластеров: {} '.format(len(unique)))
my_dict = dict(zip(unique, counts))

print(my_dict)

keys = my_dict.keys()

i = 0
clusters = {}
for key in keys:
    if my_dict[key] >= 15:
        i += 1
        # print('#{}. Кластер "{}" : {}'.format(i, key, my_dict[key]))
        clusters[key] = my_dict[key]

min_dists = []
office_inds = []
for n in clusters:
    point = cluster_centers[n]
    min_dist, office_ind = dist_to_nearest_office(coordinates_of_offices, point, debug=False)
    if debug:
        print('Кластер # {}\nБлижайший офис: {} ({})'.format(n, cities_offices[office_ind], min_dist))
    min_dists.append(min_dist)
    office_inds.append(office_ind)

# min_dists_sort = np.sort(min_dists)[:20]
min_dists_sort = np.sort(min_dists)[:5]
# print(min_dists_sort)
# print(min_dists[:5])

print('\n\t\t\tРЕЗУЛЬТАТЫ:\n')
for i, dist in enumerate(min_dists_sort):
    print('{}-е место:'.format(i+1))
    n = min_dists.index(dist)
    print('\tКластер №{} (X = {}, Y = {})'.format(n, cluster_centers[n][0], cluster_centers[n][1]))
    if i == 0:
        answer = '{} {}'.format(cluster_centers[n][0], cluster_centers[n][1])
        write_answer(answer)
        with open('coordinates.txt', "w") as fout:
            True
    city = cities_offices[office_inds[n]]
    X = coordinates_of_offices[office_inds[n]][0]
    Y = coordinates_of_offices[office_inds[n]][1]
    print('\t\tБлижайший офис: {} (X = {}, Y = {})'.format(city, X, Y))
    print('\t\tРасстояние: {}'.format(dist))
    new_line = '{},{}\n'.format(cluster_centers[n][0], cluster_centers[n][1])
    with open('coordinates.txt', "a") as fout:
        fout.write(new_line)




