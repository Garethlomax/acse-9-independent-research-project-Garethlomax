# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 02:03:17 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import random
from sklearn.neighbors import LocalOutlierFactor

def map_plot_func(test_array, vmin = 0 , vmax = 1, colour = 'viridis', border_colour = 'black'):
    """Plots PlateCarree map projection of given PRIO grid global representation

    Plots heatmap of given global array of PRIO grid encoded predictors. Plotted
    on PlateCaree projection using cartopy.

    Parameters
    ----------
    test_array: array
        input array representation of conflict parameters to display. Input must
        be of dimensions (360 x 720). Input easily constructed using
        construct_layer function.
    vmin, vmax: float, optional
        Control the cutoff of values represented in the heatmap. Passed as a parameter
        to maplotlib.pyplot.pcolormesh
    colour: str, optional.
        Optional specification of the plotting colourscheme. may be any matplotlib
        supported colorscheme.
    border_color: str, optional
        Specifies the colour of the country outlines in the plot. may be 'black',
        'white', or 'grey'.
    """

    north = 37.32
    south = -34.5115
    west = -17.3113
    east = 51.2752
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())


    y = np.arange(-90,90,0.5)
    x = np.arange(-180,180,0.5)
    xx, yy = np.meshgrid(x,y)



    ax.coastlines(color = border_colour)
    ax.add_feature(cfeature.BORDERS, edgecolor='black')

    #plot box around africa
    loc_b = [north, north, south, south, north]
    loc_a = [west, east, east, west, west]
    ax.plot(loc_a, loc_b, transform = ccrs.PlateCarree())

    cmap = cm.get_cmap(name = colour)
    ax.pcolormesh(xx, yy, test_array,vmin = vmin, vmax = vmax, transform = ccrs.PlateCarree(), cmap = cmap)
    plt.show()

