# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:44:55 2019

@author: Gareth
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs

#data = pd.read_csv("data/ged191.csv")

data = pd.read_csv("data/ged191.csv")
#data = pd.read_csv("data/PRIO-GRID Static Variables - 2019-07-26.csv")
test_array = np.zeros((360,720), dtype = np.int64)
#plt.imshow(test_array)
#x = data["xcoord"]
#y = data["ycoord"]
#z = data["cmr_max"]
plt.close()


def index_return(ind, x_dim, y_dim):
    """just for converting indices quickly"""
    x_out = ind % x_dim
    y_out = int(ind / x_dim)
    return y_out, x_out



#def parameter_grid_full(param_list):
#    """returns numpy gird of parameters
#
#    param_list is list of string with keys """
#
#    array = np.zeros((len(param_list), 360, 720))
#
#    data = pd.read_csv("data/PRIO-GRID Static Variables - 2019-07-26.csv")
#
#    for i in param_list:
#        #
#        for j in data[i]:
#            y, x = index_return(j-1,719,719)
##            array[ya, xa] =
#
#
#
#
#class data_loader():
#    def __init__(self, param_list, time_step_period):
#        self.data = pd.read_csv("data/PRIO-GRID Static Variables - 2019-07-26.csv")
#        self.param_list = param_list
#        self.array = np.zeros((len(self.param_list),360, 720))
#
#    def construct_array(self):

t1 = data.priogrid_gid
t2 = data.deaths_civilians
#t1 = data.gid


k = 0
# now we iterate over the ensemble of events
for i in range(len(t1)):
    ya, xa = index_return(t1[i]-1, 720, 360)
    test_array[ya, xa] += t2[i]
    k+= 1
print(k)

y = np.arange(-90,90,0.5)
x = np.arange(-180,180,0.5)
xx, yy = np.meshgrid(x,y)

xx = np.fliplr(xx)
xx = np.flipud(xx)
yy = np.fliplr(yy)
yy = np.flipud(yy)


#for i in data['priogrid_gid']:
#    ya, xa = index_return(i - 1, 719, 719)
#    test_array[ya, xa] += 1
#    k+= 1
#print(k)

#plt.plot(x,y,'.')
#plt.show()
ax = plt.axes(projection=ccrs.PlateCarree())
##
###plt.contourf(y, x, z, 60,
###             transform=ccrs.PlateCarree())
##
##
test_array = np.fliplr(test_array)
test_array = np.flipud(test_array)
#
###plt.contourf(test_array, transform = ccrs.PlateCarree())
###rotated_pole = ccrs.crs
###plt.imshow(x,y,z, transform = ccrs.PlateCarree())
##
#plt.pcolormesh(test_array, transform = ccrs.PlateCarree())
###


#plt.imshow(test_array)

ax.coastlines()
rp = ccrs.RotatedPole()
ax.pcolormesh(xx, yy, test_array,vmin = 0, vmax = 1, transform = ccrs.PlateCarree())

north = 37.32
south = -34.5115
west = -17.3113
east = 51.2752

loc_list = [[north,west ],[north, east],[south, east], [south, west], [north, west]]
loc_b = [north, north, south, south, north]
loc_a = [west, east, east, west, west]
ax.plot(loc_a, loc_b, transform = ccrs.PlateCarree())

def round(i):
    """for rounding - always rounding down."""
    j = int(i)
    k = i - j
    if k>0.5: # if above i.5 orrigionally
        j += 0.5
    return j

# check lattitude and longitude
def coord_to_grid(long, lat, x_dim= 720, y_dim = 360):
    """returns grid location for given grid size"""
    lat_dummy = np.arange(-90,90,0.5)
    long_dummy = np.arange(-180,180,0.5)

    round_long = round(long)
    round_lat = round(lat)

    long = np.where(long_dummy == round_long)
    lat = np.where(lat_dummy == round_lat)
    return long[0][0], lat[0][0]








##plt.contourf(test_array, vmin = 0, vmax = 1)
#
ax.coastlines()
###
plt.show()





