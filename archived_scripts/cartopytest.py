# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:11:50 2019

@author: Gareth
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/ged191.csv")

ax = plt.axes(projection=ccrs.PlateCarree())

states_provinces = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces_lines",
    scale="50m",
    facecolor="none",
)

SOURCE = "Natural Earth"
LICENSE = "public domain"
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)

# ax.stock_img()


#
# ax.add_feature(states_provinces, edgecolor='gray')
# ax.coastlines()
# ax.borders()
# ax.stock_img()

# ax.plot(data.longitude, data.latitude,'.', transform = ccrs.PlateCarree())

ax.contourf(data.longitude, data.latitude, data.best, transform=ccrs.PlateCarree())
plt.show()
