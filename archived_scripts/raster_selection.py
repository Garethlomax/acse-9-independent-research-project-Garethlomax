# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:52:15 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import h5py


"""SAVE THIS FOR JUSTIFYING SELECTION CRITERIA"""


def raster_test(input_data, chunk_size=16):
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
        for j in range(width - chunk_size + 1):
            print(input_data[:, i : i + chunk_size, j : j + chunk_size])
            print(".")


#    plt.imshow(input_data)


def raster_selection(input_data, chunk_size=16):
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
        for j in range(width - chunk_size + 1):
            input_data[0][i : i + chunk_size, j : j + chunk_size]

    plt.imshow(input_data)


def random_pixel_bounds(i, j, chunk_size=16):
    # returns the bounds of the image to select with a random pixel size.

    height = random.randint(0, chunk_size - 1)
    width = random.randint(0, chunk_size - 1)
    # this randomly generates a of the image for where the pixel may be located
    # randomly in the cut out image.
    i_lower = i - height
    i_upper = i + (chunk_size - height)

    j_lower = j - width
    j_upper = j + (chunk_size - width)

    return [i_lower, i_upper, j_lower, j_upper]


def random_selection(image, i, j, chunk_size=16):

    i_lower, i_upper, j_lower, j_upper = random_pixel_bounds(
        i, j, chunk_size=chunk_size
    )

    print(image[i_lower:i_upper, j_lower:j_upper])


def random_grid_selection(image, sequence_step, chunk_size=16, draws=5, debug=True):
    if debug:
        print("Image shape is:", image.shape)

    # decide if this is going to be h5py loaded.

    # decide what sequence step is going to be like and how to return it

    # image is seq, channels, height, width

    # here we are using a seq length of 10. - could use 12 but atm we go for 10.

    # sequence step is the step at which the TRUTH is being extracted. the predictor sequence
    # is extracted from the 10 preceding steps. be careful to send in from i > 11
    assert sequence_step > 10, (
        "This function selects the datapoints from this test set that contain"
        "a conflict event and then selects predictor data from the 10 preceding steps"
        " as a result i > 10 must be true"
    )
    # for sequence step, 0th layer - i.e fatalities
    y, x = np.where(image[sequence_step][0] >= 1)

    if debug:
        print(x.shape)

    truth_list = []
    predictor_list = []

    # re arange for loops for speed?
    for i, j in zip(y, x):  # now over sites where fatalities have occured
        for _ in range(draws):
            i_lower, i_upper, j_lower, j_upper = random_pixel_bounds(
                i, j, chunk_size=chunk_size
            )

            # now need to work out how to store these. how to stack ontop ect.
            truth = image[sequence_step][0, i_lower:i_upper, j_lower:j_upper]
            # check these dimensions
            predictors = image[i - 10 : i, :, i_lower:i_upper, j_lower:j_upper]

            truth_list.append(truth)
            predictor_list.append(predictors)

    # finally we combine the previous arrays.

    return np.stack(predictor_list, axis=0), np.stack(truth_list, axis=0)


def full_dataset_numpy(image, chunk_size=16, draws=5, debug=False):
    # image is seq, channels, height, width
    predictor_list = []
    truth_list = []
    for i in range(11, len(image)):
        t1, t2 = random_grid_selection(image, i)
        predictor_list.append(t1)
        truth_list.append(t2)

    truth_np = np.concatenate(truth_list, axis=0)
    predictor_np = np.concatenate(predictor_list, axis=0)
    return predictor_np, truth_np


def quick_dataset(data, name):
    f = h5py.File(name + ".hdf5", "w")
    f.create_dataset("main", data=data)
    #    f.create_dataset("truth", data = truth)
    f.close()


# data = pd.read_csv("data/ged191.csv")


def debug_func1(dataframe, month):
    a = dataframe[dataframe.mon_month == month]
    print(len(a[a.best > 0]))


def full_dataset_h5py(image, filename, chunk_size=16, draws=5, debug=False):
    """ dataset is too large to combine in 12gb of ram - need to combine in h5py
    array. i.e lazy saving as well as lazy loading"""

    f = h5py.File(filename + ".hdf5", "w")
    for i in range(11, len(image)):
        t1, t2 = random_grid_selection(image, i)
        if i == 11:
            # creat h5py file at first step.
            f.create_dataset(
                "predictor", data=t1, maxshape=(None,)
            )  # compression="gzip", chunks=True, taken out
            f.create_dataset("truth", data=t2, maxshape=(None,))

        else:
            f["predictor"].resize(
                (f["predictor"].shape[0] + t1.shape[0]), axis=0
            )  # expand dataset
            f["truth"].resize((f["truth"].shape[0] + t2.shape[0]), axis=0)

            f["predictor"][-t1.shape[0] :] = t1  # place new data in expanded dataset
            f["truth"][-t2.shape[0] :] = t2

    f.close()


# raster_test(test_raster, 3)

# def
