# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:52:15 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt
import random
test_raster = np.arange(0, 20, 1)
#test_raster = np.zeros(())
test_raster.resize((4,5))

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
            print(input_data[:,i:i+chunk_size,j:j + chunk_size])
            print(".")

#    plt.imshow(input_data)

def raster_selection(input_data, chunk_size = 16):
    # here input_data is sequence step.
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
    # this is not efficient.
    for i in range(height - chunk_size + 1):
        for j in range(width - chunk_size+1):
            input_data[0][i:i+chunk_size,j:j + chunk_size]

    plt.imshow(input_data)


def random_pixel_bounds(i, j, chunk_size = 16):
    # returns the bounds of the image to select with a random pixel size.

    height = random.randint(0, chunk_size-1)
    width = random.randint(0, chunk_size-1)
    # this randomly generates a of the image for where the pixel may be located
    # randomly in the cut out image.
    i_lower = i - height
    i_upper = i + (chunk_size - height)

    j_lower = j - width
    j_upper = j + (chunk_size - width)

    return [i_lower, i_upper, j_lower, j_upper]

def random_selection(image, i, j, chunk_size = 16):

    i_lower, i_upper, j_lower, j_upper = random_pixel_bounds(i, j, chunk_size = chunk_size)

    print(image[i_lower:i_upper,j_lower:j_upper])












raster_test(test_raster, 3)

#def


