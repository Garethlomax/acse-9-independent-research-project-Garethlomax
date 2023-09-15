# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:12:32 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py

test_dat = np.zeros(10)
with h5py.File("simple_attribute_test4.hdf5", "w") as t:

    t.create_dataset("main", data=test_dat)
    key_list = [u"petroleum_y", u"drug_y", u"prec_gpcp"]
    t["main"].attrs.create("key_prio", np.string_(key_list))


def find_avg_lazy_load(filename):
    # file name is name of hdf5 file
    data = h5py.File(filename, "r")
    predictor = data["predictor"]
    channel_num = predictor.shape[2]
    avg = np.zeros(channel_num)
    std = np.zeros_like(avg)
    for i in range(channel_num):
        avg = np.avg(predictor[:, :, i])
        std = np.avg(predictor[:, :, i])
    return avg, std
