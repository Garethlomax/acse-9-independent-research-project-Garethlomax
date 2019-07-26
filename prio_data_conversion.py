# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:44:55 2019

@author: Gareth
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#data = pd.read_csv("data/ged191.csv")
data = pd.read_csv("data/PRIO-GRID Static Variables - 2019-07-26.csv")

test_array = np.zeros((360,720), dtype = np.int64)
#plt.imshow(test_array)
x = data["xcoord"]
y = data["ycoord"]
z = data["cmr_max"]


def index_return(ind, x_dim, y_dim):
    """just for converting indices quickly"""
    x_out = ind % x_dim
    y_out = int(ind / y_dim)
    return y_out, x_out



k = 0
# now we iterate over the ensemble of events
for i in data['gid']:
    ya, xa = index_return(i - 1, 719, 719)
    test_array[ya, xa] += 1
    k+= 1
print(k)

#plt.plot(x,y,'.')
#plt.show()
import cartopy.crs as ccrs
ax = plt.axes(projection=ccrs.PlateCarree())
##
##plt.contourf(y, x, z, 60,
##             transform=ccrs.PlateCarree())
#
#
test_array = np.fliplr(test_array)
test_array = np.flipud(test_array)
#
##plt.contourf(test_array, transform = ccrs.PlateCarree())
##rotated_pole = ccrs.crs
##plt.imshow(x,y,z, transform = ccrs.PlateCarree())
#
plt.pcolormesh(x, y, test_array, transform = ccrs.PlateCarree())
#
ax.coastlines()
#
plt.show()