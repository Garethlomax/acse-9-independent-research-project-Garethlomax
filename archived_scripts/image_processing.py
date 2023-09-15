# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:48:15 2019

@author: Gareth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py


data_prio = pd.read_csv(
    "data/PRIO-GRID Yearly Variables for 1946-2014 - 2019-07-26.csv"
)
data_ucdp = pd.read_csv("data/ged191.csv")

#
# test_array_2 = np.zeros(360*720)
# for i in range(len(t1)):
#    j = t1.iloc[i] - 1
#    test_array_2[j] += t2.iloc[i]
#
# test_array_2 = test_array_2.reshape(360,720)

# def date_to_int_list(dates):
#    # date is in format yyyy-mm-dd
#    for date in dates:
#        y = int(date[0:4])
#        m = int(date[5:7])
#        d = int(date[8:10])
#
##    print("date is:")
##    print(y, " ",m, " ", d)
#    return [y,m,d]
#
# def monotonic_date(date, baseline = [1989, 1, 1]):
#    #returns date as monotinic functikn of months passed since.
#    date = date_to_int_list(date)
##    print(type(date[0]))
##    print(type(baseline[0]))
#    # turns date since baseline start date into a monotonic function based on
#    # year and months in line with pgm unit of analysis
#    return date[1] - baseline[1] + ((date[0] - baseline[0]) * 12)
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


def binary_event_column(dataframe):
    """ as it is hard to encode categorical information about battles when
    battle deaths = 0, we then encode a binary - 0, 1 layer pertaining to
    whether an event took place"""
    new_col = np.ones(len(dataframe))
    dataframe["binary_event"] = new_col


def nan_to_one(dataframe, key):
    """takes column from dataframe"""
    dataframe[key] = dataframe[key].fillna(0)


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


key_list_prio = [
    "gcp_ppp",
    "petroleum_y",
    "drug_y",
    "prec_gpcp",
]  # not temp - needs better imputation
# excluded also useful - talk to nils about exclusion.

# no mountainous regions so far
