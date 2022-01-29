import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
# import tensorflow as tf
# print(tf.__version__)


points = ['A', 'B', 'C']
points1 = list(points)
track = []

def func(points, track=[], tracks=[]):
    for p in points:
        print('\t'*len(track), p)
        track.append(p)
        new_points = list(points)
        new_points.remove(p)
        # print('new_points = ', new_points, ' | ', end='\t')
        # print('len(track) = ', len(track))

        if len(new_points) == 0:
            print('track = ', track)
            tracks.append(track)
            print('tracks = ', tracks)
            # track = track[:-2]
            # print(f'func({new_points}) - из if')
            # func(new_points, ['Aa'])
            # func(new_points)
            return tracks
        # print(f'func({new_points}, {track}) - штатно')
        func(new_points, track, tracks)
        # track = track[:-2]

# res = func(points)
# print(res)
# print(res)
tracks=[]
for p1 in points1:
    print('\t'*len(track), p1)
    track.append(p1)
    points2 = list(points1)
    points2.remove(p1)
    for p2 in points2:
        print('\t'*len(track), p2)
        track.append(p2)
        points3 = list(points2)
        points3.remove(p2)
        for p3 in points3:
            print('\t'*len(track), p3)
            track.append(p3)
            points4 = list(points3)
            points4.remove(p3)
            for p4 in points4:
                print('\t'*len(track),  p4)
                track.append(p4)
                points5 = list(points4)
                points5.remove(p4)
                print(track)
                tracks.append(track)
            track = track[:-2]
        track = track[:-1]
    track = track[:-1]
    # track = []
print(tracks)