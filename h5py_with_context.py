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
with h5py.File("simple_attribute_test3.hdf5",'w') as t:

    t.create_dataset("main", data = test_dat)
    key_list = [u"gcp_ppp", u"petroleum_y",u"drug_y", u"prec_gpcp"]
    t["main"].attrs.create("key_prio", np.string_(key_list))