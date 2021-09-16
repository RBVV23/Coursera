import numpy as np
import itertools

permutations = itertools.permutations([0, 1, 2])
# print(list(permutations))

# for a,b,c in permutations:
#     print('a = {}, b = {}, c = {}'.format(a,b,c))
#     mapping = mapping = {2:a, 1:b, 0:c}
#     print(mapping)



def dist_to_nearest_office(offices, cluster_center, debug=False):
    dists = []
    if debug:
        print('Center: X = {}; Y = {}'.format(cluster_center[0], cluster_center[1]))
    for item in offices:
        dist = ((item[0] - cluster_center[0])**2 + (item[1] - cluster_center[1])**2 )**0.5
        dists.append(dist)
        if debug:
            print('\tOffice: X = {}; Y = {}'.format(item[0], item[1]))
            print('\tDistantion = ', dist)
    min_dist = min(dists)
    ind = np.argmin(dist)
    if debug:
        print('\t\tNearest office #{}: {}'.format(ind, offices[ind]))
        print('\t\tMinimal distance: {}'.format(min_dist))
    return min_dist, offices[ind]
#
#
# offices = [[-10, 10], [20, -20], [-30, -30]]
# cluster_centers = [[5, 5], [10,10], [0,0]]
# # dist, office = dist_to_nearest_office(offices, cluster_centers[0], True)
# # print('dist: {}, office: {}'.format(dist, office))
#
# dists = []
# for point in cluster_centers:
#     dist, office = dist_to_nearest_office(offices, point)
#     dists.append([dist, office])
# print(dists)


coordinates_of_offices = [[33.751277, -118.188740],
                          [25.867736, -80.324116],
                          [51.503016, -0.075479],
                          [52.378894, 4.885084],
                          [39.366487, 117.036146],
                          [-33.868457, 151.205134]]

cluster_center = [32.72532479999998, -114.62439700000003]
dist_to_nearest_office(coordinates_of_offices, cluster_center, debug=True)