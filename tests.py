# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:15:03 2019

@author: Gareth
"""


import hpc_construct as cf
import latest_run as lr
import pandas as pd
import numpy as np
import random
import h5py
from isolated_test_class import *


def test_date_to_int_list():
    date = "2019-10-10"
    assert cf.date_to_int_list(date) == [2019, 10, 10], "date_to_int_list failed"


def test_monotonic_date():
    date = "1989-02-01"
    assert cf.monotonic_date(date) == 1, "failed"


# def test_construct_layer()


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
    i_low, i_high, j_low, j_high = cf.random_pixel_bounds(0, 0, 16)
    assert (abs(i_low) + abs(i_high)) == 16, "i dimension wrong"
    assert (abs(j_low) + abs(j_high)) == 16, "j dimension wrong"


def test_random_grid_selection():
    """Test of random_grid_selection function
    """
    # dummy input image
    dummy_image = np.zeros([20, 1, 360, 720])
    dummy_image[:, :, 100, 100] = 1

    # set value to pick out.
    pred_list, truth_list = cf.random_grid_selection(
        dummy_image, 19, draws=5, debug=False
    )
    assert (len(pred_list) == 5) and (
        len(truth_list) == 5
    ), "not providing correct number of image samples"
    assert (
        np.sum(truth_list) == 5
    ), "as one event per layer to extract from, we expect each truth image to contain 1 event"


def test_full_dataset_h5py():
    """Test of full dataset creation

    Dummy dataset is saved then reloaded to test h5py testing.
    """
    filename = "test_dataset"
    dummy_image = np.zeros([20, 1, 360, 720])
    dummy_image[:, :, 100, 100] = 1

    cf.full_dataset_h5py(
        dummy_image,
        filename,
        key_list_prio="prio_test",
        key_list_ucdp="ucdp_test",
        min_events=0,
    )

    with h5py.File(filename + ".hdf5", "r") as f:
        assert len(f["truth"]) == 45, "dataset contains wrong number of samples"
        assert f["truth"].attrs["key_ucdp"] == b"ucdp_test", "meta data not saving"


#        assert


def test_find_avg_lazy_load():
    """Test of full dataset creation

    Uses test dataset produced in the previous test. As this is small enough to
    fit into memory we can find the averages and verify when too large.
    """
    f = h5py.File("test_dataset.hdf5", "r")
    avg, std = cf.find_avg_lazy_load(f)
    assert avg[0] == np.average(f["predictor"])
    assert std[0] == np.std(f["predictor"])


def test_LSTMunit_autograd():
    """Tests end to end differentiability of LSTMunit_t.
    """
    shape = [1, 1, 16, 16]
    x = torch.zeros(shape, dtype=torch.double, requires_grad=True)
    h = torch.zeros([1, 2, 16, 16], dtype=torch.double, requires_grad=True)
    c = torch.zeros([1, 2, 16, 16], dtype=torch.double, requires_grad=True)
    testunit = LSTMunit_t(1, 2, 3).double()
    torch.autograd.gradcheck(testunit, (x, h, c), eps=1e-4, raise_exception=True)


def test_LSTMencdec():
    """Tests end to end differentiability of LSTMencdec
    """
    structure = np.array([[2, 4, 0], [0, 4, 2]])
    encdec = LSTMencdec_onestep_t(structure, 1, 5)
    shape = [1, 10, 1, 16, 16]
    x = torch.zeros(shape, dtype=torch.double, requires_grad=True)
    torch.autograd.gradcheck(encdec, (x,), eps=1e-4, raise_exception=True)


# test_date_column()
# test_date_to_int_list()
# test_monotonic_date()
# test_construct_layer()
# test_binary_event_column()
# test_nan_to_one()
# test_random_pixel_bounds()
# test_random_grid_selection()
# test_full_dataset_h5py()
# test_find_avg_lazy_load()
