# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:26:42 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/ged191.csv")

hist = np.bincount(data.priogrid_gid)
plt.figure()
plt.plot(hist)
plt.title("histogram of gid - can clearly see high intensity zones")



# data reading in


