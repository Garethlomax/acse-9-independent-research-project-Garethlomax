# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:52:15 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt

#test_raster = np.arange(0, 64, 1)
test_raster = np.zeros((200,200))
#test_raster.resize((8,8))

"""SAVE THIS FOR JUSTIFYING SELECTION CRITERIA"""

def raster_test(input_data, chunk_size = 16):
    # to overcome edge sizes can make selection large if we just reject the training data for outside africa
    # although we do not necessarily need to do this
    # i.e expand box and allow less sampled box to sampel others more frequently.

    # step size is always 1
    # assuming image is a cutout of globe
    # this is for single step, single channel as a test.
    step = 1
    height = input_data.shape[-2]
    width = input_data.shape[-1]
    for i in range(height - chunk_size + 1):
        for j in range(width - chunk_size+1):
            input_data[i:i+chunk_size,j:j + chunk_size] += 1

    plt.imshow(input_data)

def raster_selection(input_data, chunk_size = 16):
    # data should be of dimensions seq, channels, height, width.
    # to overcome edge sizes can make selection large if we just reject the training data for outside africa
    # although we do not necessarily need to do this
    # i.e expand box and allow less sampled box to sampel others more frequently.

    # step size is always 1
    # assuming image is a cutout of globe
    # this is for single step, single channel as a test.
    step = 1
    height = input_data.shape[-2]
    width = input_data.shape[-1]
    for i in range(height - chunk_size + 1):
        for j in range(width - chunk_size+1):
            input_data[i:i+chunk_size,j:j + chunk_size] += 1

    plt.imshow(input_data)






raster_selection(test_raster, 16)

#def


