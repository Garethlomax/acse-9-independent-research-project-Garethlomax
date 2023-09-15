# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:09:54 2019

@author: Gareth
"""
# import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py
import sklearn

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import AgglomerativeClustering, MeanShift, AffinityPropagation
from sklearn.neighbors import LocalOutlierFactor

from hpc_construct import *


data = pd.read_csv("data/ged191.csv")

north = 37.32
south = -34.5115
west = -17.3113
east = 51.2752

# africa = data[]

data = data[
    (data.latitude >= south)
    & (data.latitude <= north)
    & (data.longitude >= west)
    & (east >= data.longitude)
]
# subsampling data to africa data.

africa = data[data.region == "Africa"]
africa_2000 = africa[africa.year == 2000]

# construct layer here

l = construct_layer(africa_2000, key="best", prio_key="priogrid_gid")

y, x = np.array(np.where(l > 0))
X = []
for i in range(len(x)):
    X.append([x[i], y[i]])

X = np.array(X)


clustering = AgglomerativeClustering(n_clusters=20).fit(X)
mshift = MeanShift(cluster_all=False, min_bin_freq=5).fit(X)
af = AffinityPropagation(damping=0.8).fit(X)
# ax.plot(x, y, '.',c = clustering.labels_,transform = ccrs.PlateCarree() )
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.05)

# ax = plt.axes(projection=ccrs.PlateCarree())
#
# colors = ['red','green','blue','purple']
##fig = plt.figure(figsize=(8,8))
# fig, ax = plt.subplots(1, 1, figsize=(64, 8))
#
# ax.scatter(X[:,0], X[:,1], c=list(clustering.labels_))
#
# fig, ax = plt.subplots(1, 1, figsize=(64, 8))
#
# ax.scatter(X[:,0], X[:,1], c=list(mshift.labels_))
#
# fig, ax = plt.subplots(1, 1, figsize=(64, 8))
#
# ax.scatter(X[:,0], X[:,1], c=list(af.labels_))


y_pred = clf.fit_predict(X)

fig, ax = plt.subplots(1, 1, figsize=(64, 8))

ax.scatter(X[:, 0], X[:, 1], c=list(y_pred))


for i in range(len(x)):
    X.append([x[i], y[i]])

X = np.array(X)
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.05)
y_pred = clf.fit_predict(X)
bools = y_pred == 1
non_outs = X[vals]
x = non_outs[:, 0]
y = non_outs[:, 1]

# ax.plot(x, y, '.',c = clustering.labels_,transform = ccrs.PlateCarree() )

# ax.coastlines()

#    loc_list = [[north,west ],[north, east],[south, east], [south, west], [north, west]]

# ax.plot(africa_2000.longitude, africa_2000.latitude,'.', transform = ccrs.PlateCarree())


plt.show()
