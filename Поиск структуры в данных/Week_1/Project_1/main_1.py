import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

def write_answer(answer):
    name = 'answer.txt'
    with open(name, "w") as fout:
        fout.write(str(answer))

def dist_to_nearest_office(offices, cluster_center, debug=False):
    if debug:
        print()
        print('\tCenter: X = {}; Y = {}'.format(cluster_center[0], cluster_center[1]))
    dists = []
    for item in offices:
        dist = ((item[0] - cluster_center[0])**2 + (item[1] - cluster_center[1])**2 )**0.5
        dists.append(dist)
        if debug:
            print('\t\tOffice: X = {}; Y = {}'.format(item[0], item[1]))
            print('\t\tDistantion = ', dist)
    min_dist = min(dists)
    ind = dists.index(min_dist)
    if debug:
        print('\t\t\tNearest office #{}: {}'.format(ind, offices[ind]))
        print('\t\t\tMinimal distance: {}'.format(min_dist))
    return min_dist, ind

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 100)

debug=False

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
data = small_data

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

# print(my_dict)

keys = my_dict.keys()

i = 0
clusters = {}
clusters_list = []
for key in keys:
    if my_dict[key] >= 15:
        clusters[key] = my_dict[key]
        clusters_list.append([i, key])
        i += 1

min_dists = []
office_inds = []
for n, key in clusters_list:
    point = cluster_centers[key]
    min_dist, office_ind = dist_to_nearest_office(coordinates_of_offices, point, debug=debug)
    if debug:
        print('Кластер # {}\nБлижайший офис: {} ({})'.format(n, cities_offices[office_ind], min_dist))
    min_dists.append(min_dist)
    office_inds.append(office_ind)


min_dists_sort = np.sort(min_dists)[:20]

print('\n\t\t\tРЕЗУЛЬТАТЫ:\n')
for i, dist in enumerate(min_dists_sort):
    n = min_dists.index(dist)
    print('{}-е место: (id={})'.format(i+1, n))
    key = clusters_list[n][1]
    print('\tКластер №{} (X = {}, Y = {})'.format(n, cluster_centers[key][0], cluster_centers[key][1]))
    if i == 0:
        answer = '{} {}'.format(cluster_centers[key][0], cluster_centers[key][1])
        write_answer(answer)
        with open('coordinates.txt', "w") as fout:
            pass
    city = cities_offices[office_inds[n]]
    X = coordinates_of_offices[office_inds[n]][0]
    Y = coordinates_of_offices[office_inds[n]][1]
    print('\t\tБлижайший офис: {} (X = {}, Y = {})'.format(city, X, Y))
    print('\t\tРасстояние: {}'.format(dist))
    new_line = '{},{}\n'.format(cluster_centers[key][0], cluster_centers[key][1])
    with open('coordinates.txt', "a") as fout:
        fout.write(new_line)




