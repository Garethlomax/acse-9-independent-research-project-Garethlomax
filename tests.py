# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:15:03 2019

@author: Gareth
"""

"""By default pytest only identifies the file names starting with test_ or ending
 with _test as the test files. We can explicitly mention other filenames though (explained later).
 Pytest requires the test method names to start with test.
 All other method names will be ignored even if we explicitly ask to run those methods."""

import collated_funcs as cf
import pandas as pd
import numpy as np
import random

def test_date_to_int_list():
    date = "2019-10-10"
    assert cf.date_to_int_list(date) == [2019,10,10], "date_to_int_list failed"

def test_monotonic_date():
    date = "1989-02-01"
    assert cf.monotonic_date(date) == 1, "failed"

#def test_construct_layer()



def test_date_column():
    dummy_dataframe = pd.DataFrame()
    dummy_dataframe["date_start"] = ["1989-02-01"]
    cf.date_column(dummy_dataframe)
    assert dummy_dataframe["mon_month"][0] == 1, "failed"

def test_construct_layer():
    dummy_dataframe = pd.DataFrame()
    dummy_dataframe["gid"] = [200]
    dummy_dataframe["dummy"] = [1]
    l = cf.construct_layer(dummy_dataframe, "dummy")
    loc_a, loc_b = np.where(l == 1)
    assert loc_a[0] == 0, "failed"
    assert loc_b[0] == 199, "failed"


def test_binary_event_column():
    dummy_dataframe = pd.DataFrame()
    dummy_dataframe["gid"] = [200]
    dummy_dataframe["dummy"] = [1]
    cf.binary_event_column(dummy_dataframe)
    assert dummy_dataframe["binary_event"][0] == 1, "failed"

def test_nan_to_one():
    dummy_dataframe = pd.DataFrame()
    dummy_dataframe["dummy"] = [np.NaN]
    cf.nan_to_one(dummy_dataframe, "dummy")
    assert dummy_dataframe["dummy"][0] == 0, "failed"

def test_random_pixel_bounds():
    i_low, i_high, j_low, j_high = cf.random_pixel_bounds(0,0,16)
    assert (abs(i_low)+ abs(i_high))==16, "i dimension wrong"
    assert (abs(j_low)+ abs(j_high))==16, "j dimension wrong"










