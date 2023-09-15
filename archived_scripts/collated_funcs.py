# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:14:32 2019

@author: Gareth
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import h5py


def mod_dif(module_1, module_2):
    print(set(dir(module_1)) - set(dir(module_2)))


def mod_dif_full(m1, m2, m3):
    m1 = set(dir(m1))
    m2 = set(dir(m2))
    m3 = set(dir(m3))
    print((m1 - m2) - m3)


def lists_overlap(a, b):
    for i in a:
        if i in b:
            print(i)


def mod_overlap(module_1, module_2):
    lists_overlap(dir(module_1), dir(module_2))


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


def construct_layer(dataframe, key, prio_key="gid", debug=False):
    # returns 360 720 grid layer for a given parameter
    # should be given for one parameter per year.
    array = np.zeros(360 * 720)
    prio_grid = dataframe[prio_key]
    for i in range(len(prio_grid)):
        j = prio_grid.iloc[i] - 1
        """change the below to only be in the case it isnt nan"""  # will this cause problems?
        array[j] += dataframe[key].iloc[i]
    array.resize(360, 720)

    return array


def construct_combined_sequence(
    dataframe_prio,
    dataframe_ucdp,
    key_list_prio,
    key_list_ucdp,
    start=[1989, 1, 1],
    stop=[2014, 1, 1],
):
    stop = "2014-01-01"
    # need to adapt ged and other year / month vs ged database for this.
    # bool prio to add multiples of 12 to each year usin prio grid.
    num_month = monotonic_date(stop, start)
    print(num_month)
    comb_channel_len = len(key_list_prio) + len(key_list_ucdp)
    print(comb_channel_len)
    array = np.zeros((num_month, comb_channel_len, 360, 720))

    stop = date_to_int_list(stop)

    month = 0
    for i in range(start[0], stop[0]):
        for j in range(12):  # for each month
            # now fill in selected channels as requried.

            array[month][: len(key_list_ucdp)] = construct_channels(
                dataframe_ucdp[dataframe_ucdp.mon_month == month],
                key_list=key_list_ucdp,
                prio_key="priogrid_gid",
            )
            array[month][len(key_list_ucdp) :] = construct_channels(
                dataframe_prio[dataframe_prio.year == i],
                key_list=key_list_prio,
                prio_key="gid",
            )
            print(month)

            month += 1
    del month
    return array


def construct_channels(dataframe, key_list, prio_key="gid"):
    # usually used for prio
    array = np.zeros((len(key_list), 360, 720))
    for i, keys in enumerate(key_list):
        array[i] = construct_layer(dataframe, key=keys, prio_key=prio_key)
    return array


"""check how we are dealing with cases that go up to and including the final step"""


def construct_sequence(
    dataframe,
    key_list,
    prio_key="gid",
    start=[1989, 1, 1],
    stop=[2014, 1, 1],
    prio=False,
):
    stop = "2014-01-01"
    # need to adapt ged and other year / month vs ged database for this.
    # bool prio to add multiples of 12 to each year usin prio grid.
    num_month = monotonic_date(stop, start)

    if prio == False:
        # i.e if doing ucdp
        # presumes adapted ucdp
        # seq length, channels, height, width
        array = np.zeros((num_month, len(key_list), 360, 720))

        for month in range(num_month):
            array[month] = construct_channels(
                dataframe[dataframe.mon_month == month],
                key_list=key_list,
                prio_key="priogrid_gid",
            )

            # now for
    #            array[i] =
    elif prio:
        stop = [2014, 1, 1]
        array = np.zeros((num_month, len(key_list), 360, 720))
        # now for the prio grid data.
        # need to make up remainder of start year,
        # then multiples of 12 for each year
        # then remainder of end year.

        """this presumes start dates @ start of year no more no less"""
        # need to plus one at the end
        month = 0
        for i in range(start[0], stop[0]):
            for j in range(12):  # for each month
                array[month] = construct_channels(
                    dataframe[dataframe.year == i], key_list=key_list, prio_key="gid"
                )
                print(month)

                month += 1
        del month

    #        start_months = 13 - start[1] # i.e if its 1989,1,1 then there are 12 months left.
    #        years = stop[0] - start[0]  - 1 # -1 as due to method of making up start months. i.e
    #        # want [2013,1,1] [2014,1,1] to be dependant of start and stop months and have no year * 12 months additions
    #        finish_months = stop[1] # stop months
    #        months_interim = np.arange(start[0], stop[0]+1, 1)
    #
    #        ######
    #
    #        # now the start month multiples
    #        for i in range(start_months):
    #            array[i] = construct_channels(dataframe[dataframe.year == start[0]], key_list = key_list, prio_key = "gid")
    #        # double check prio yearly - try to get monthly values out.
    #
    #        for i in range(years):
    #            for j in range(i *12, (i+1)* 12):
    #                array[j]

    return array


def date_column(dataframe, baseline=[1989, 1, 1]):
    # puts new column on dataframe, no need to return.
    # date start just as dummy atm
    #    dataframe = dataframe["date_start"]
    vals = dataframe["date_start"].values
    new_col = np.array([monotonic_date(string_date) for string_date in vals])
    dataframe["mon_month"] = new_col


def h5py_conversion(data_array, filename, key_list_ucdp, key_list_prio):
    # this is for saving the default 360:720 file to chop out of.
    # lazy loading saves the day
    # all day
    # every day
    f = h5py.File("{}.hdf5".format(filename), "w")

    f.create_dataset("data_combined", data=data_array)

    f.close()

    csv = open(name + "_config.csv", "w")
    csv.write("Included data UCDP:\n")
    for key in key_list_ucdp:
        csv.write(key + "\n")

    csv.write("Included data PRIO:\n")
    for key in key_list_prio:
        csv.write(key + "\n")
    csv.close()


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


def random_grid_selection(
    image,
    sequence_step,
    chunk_size=16,
    draws=5,
    cluster=False,
    min_events=0,
    debug=True,
):
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

    """INCLUDE CLUSTERING IN HERE?????"""
    if cluster:
        clf = LocalOutlierFactor(n_neighbors=5, contamination=0.05)
        pred = clf.fit_predict(X)
        # correct this but answer is this shape
        np.hstack((a.reshape((-1, 1)), b.reshape((-1, 1))))

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

            # if statement here to decide whether will be appended to full list
            # i.e if it reaches the cutoff for acceptable level of conflict.
            if np.count_nonzero(predictors[:, 0]) >= min_events:
                truth_list.append(truth)
                predictor_list.append(predictors)

    # finally we combine the previous arrays.

    return np.stack(predictor_list, axis=0), np.stack(truth_list, axis=0)


def full_dataset_numpy(image, chunk_size=16, draws=5, min_events=0, debug=False):
    # image is seq, channels, height, width
    predictor_list = []
    truth_list = []
    for i in range(11, len(image)):
        t1, t2 = random_grid_selection(image, i, min_events=min_events)
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


def binary_event_column(dataframe):
    """ as it is hard to encode categorical information about battles when
    battle deaths = 0, we then encode a binary - 0, 1 layer pertaining to
    whether an event took place"""
    new_col = np.ones(len(dataframe))
    dataframe["binary_event"] = new_col


def nan_to_one(dataframe, key):
    """takes column from dataframe"""
    dataframe[key] = dataframe[key].fillna(0)


def index_return(ind, x_dim, y_dim):
    """just for converting indices quickly"""
    x_out = ind % x_dim
    y_out = int(ind / x_dim)
    return y_out, x_out


def round(i):
    """for rounding - always rounding down."""
    j = int(i)
    k = i - j
    if k > 0.5:  # if above i.5 orrigionally
        j += 0.5
    return j


def coord_to_grid(long, lat, x_dim=720, y_dim=360):
    """returns grid location for given grid size"""
    lat_dummy = np.arange(-90, 90, 0.5)
    long_dummy = np.arange(-180, 180, 0.5)

    round_long = round(long)
    round_lat = round(lat)
    #    print(round_lat)

    long = np.where(long_dummy == round_long)
    lat = np.where(lat_dummy == round_lat)
    #    lat  = y_dim - lat[0][0]
    #    long = x_dim - long[0][0]
    lat = lat[0][0]
    long = long[0][0]
    #    print(lat)
    return long, lat


def map_plot_func(test_array, vmin=0, vmax=1, colour="viridis", border_colour="black"):

    north = 37.32
    south = -34.5115
    west = -17.3113
    east = 51.2752
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ##
    ###plt.contourf(y, x, z, 60,
    ###             transform=ccrs.PlateCarree())
    ##
    ##
    #    test_array = np.fliplr(test_array)
    #    test_array = np.flipud(test_array)

    y = np.arange(-90, 90, 0.5)
    x = np.arange(-180, 180, 0.5)
    xx, yy = np.meshgrid(x, y)

    #    ax = plt.axes(projection=ccrs.PlateCarree())
    ##
    ###plt.contourf(y, x, z, 60,map
    ###             transform=ccrs.PlateCarree())
    ##
    ##
    #    test_array = np.fliplr(test_array)
    #    test_array = np.flipud(test_array)

    ax.coastlines(color=border_colour)
    #    states_provinces = cfeature.NaturalEarthFeature(
    #        category='cultural',
    #        name='admin_1_states_provinces_lines',
    #        scale='50m',
    #        facecolor='none')
    ax.add_feature(cfeature.BORDERS, edgecolor="black")
    #    ax.
    #    loc_list = [[north,west ],[north, east],[south, east], [south, west], [north, west]]
    loc_b = [north, north, south, south, north]
    loc_a = [west, east, east, west, west]
    ax.plot(loc_a, loc_b, transform=ccrs.PlateCarree())

    cmap = cm.get_cmap(name=colour)

    ax.pcolormesh(
        xx,
        yy,
        test_array,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )

    plt.show()


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
