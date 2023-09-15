# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:26:42 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("data/ged191.csv")

hist = np.bincount(data.priogrid_gid)
plt.figure()
plt.plot(hist)
plt.title("histogram of gid - can clearly see high intensity zones")

hist = np.bincount(data.deaths_civilians[data.deaths_civilians > 0])
plt.figure()
plt.plot(hist)
plt.title("histogram of death per event")

# sns.jointplot(x="longitude", y="latitude", data=data, kind="kde");

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
# data reading in
hist = np.bincount(data.year)
plt.figure()
plt.plot(hist)
plt.title("year")


def date_to_int_list(date):
    # date is in format yyyy-mm-dd

    y = int(date[0:4])
    m = int(date[5:7])
    d = int(date[8:10])

    #    print("date is:")
    #    print(y, " ",m, " ", d)
    return [y, m, d]


def monotonic_date(date, baseline=[1989, 1, 1]):

    date = date_to_int_list(date)
    #    print(type(date[0]))
    #    print(type(baseline[0]))
    # turns date since baseline start date into a monotonic function based on
    # year and months in line with pgm unit of analysis
    return date[1] - baseline[1] + ((date[0] - baseline[0]) * 12)


# monotonic_date(data.date_start[0])
