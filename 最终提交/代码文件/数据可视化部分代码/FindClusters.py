# Imports
import pandas as pd
import datetime as dt
import time
import numpy as np


def convertToMeters(longitude, latitude):
    return [(111.03*longitude), (111.03*latitude)]


def distance(pt1, pt2):
    x = convertToMeters(abs(pt1[0]-pt2[0]))
    y = convertToMeters(abs(pt1[1] - pt2[1]))
    return np.sqrt((x**2+y**2))


def timeBetween(pt1, pt2):
    time1 = dt.datetime.strptime(pt1[2][:-3], '%Y/%m/%d %H:%M:%S')
    time2 = dt.datetime.strptime(pt2[2][:-3], '%Y/%m/%d %H:%M:%S')
    return abs(time1-time2)

def convertTime(t):
    parsedTime = dt.datetime.strptime(t[:-3], '%Y/%m/%d %H:%M:%S')
    unix = time.mktime(parsedTime.timetuple())
    return unix

def findClusters(df):
    data = df.values
    cluster = []
    clusters = []
    startPt = data[0]
    ptnum = 0

    for i in range(1, len(data)):
        if distance(startPt, data[i]):
            cluster.append(data[i])
        elif len(cluster) > 0 and timeBetween(cluster[0], cluster[-1]) > 900:
            clusters.append(cluster)
            cluster = []
    return clusters

    #while distance(startPt, data[ptnum]) < 100 and ptnum < len(data):
    #    clusters.append(data[ptnum])
    #    ptnum += 1


gps_data = pd.read_csv('Data/AA00004.csv')
columns = ['X', 'Y', 'time']
gps_data = gps_data.filter(items=columns)
#print gps_data.head()

gps_values = gps_data.values

for i in range(0, len(gps_values)):
    gps_values[i][2] = convertTime(gps_values[i][2])

gps_values = pd.DataFrame(data=gps_values)
gps_values.columns = columns
print (gps_values)

gps_values.to_csv('Data/gps_data1.csv')


gps_data = pd.read_csv('Data/AA00001.csv')
columns = ['X', 'Y']
gps_data = gps_data.filter(items=columns)

gps_values = gps_data.values
gps_values = pd.DataFrame(data=gps_values)
gps_values.columns = columns
print (gps_values)

#gps_values.to_csv('Data/gps_data2.csv')
