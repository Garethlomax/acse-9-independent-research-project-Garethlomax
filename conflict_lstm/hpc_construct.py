# -*- coding: utf-8 -*-
"""
Author: Gareth Lomax
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import h5py
import random
from sklearn.neighbors import LocalOutlierFactor


def date_to_int_list(date):
    """Transforms string date into list

    Parameters
    ----------
    date: str
        String of format 'yyyy-mm-dd'

    Returns
    -------
    list, int
        List of numerical format [y,m,d]
    """
    # date is in format yyyy-mm-dd

    y = int(date[0:4])
    m = int(date[5:7])
    d = int(date[8:10])

    #    print("date is:")
    #    print(y, " ",m, " ", d)
    return [y, m, d]


def monotonic_date(date, baseline=[1989, 1, 1]):
    """Returns the number of months elapsed between date and baseline

    Parameters
    ----------
    date: str
        Date in string format 'yyyy-mm-dd'. Input as string as this is the default
        date storage of PRIO and UCDP data formats.
    baseline: list, int
        Numerical date in format [y,m,d].

    Returns
    -------
    int:
        Elapsedc number of months between date and baseline
    """

    date = date_to_int_list(date)
    return date[1] - baseline[1] + ((date[0] - baseline[0]) * 12)


def construct_layer(dataframe, key, prio_key="gid", debug=False):
    """Constructs single global parameter map of PRIO encoded variables

    Takes input dataframe of either PRIO or UCDP data. Produces 360 x 720 array
    prio grid single layer representation of conflict predictor specified by
    key input.

    Parameters
    ----------
    dataframe: Pandas Dataframe
        Dataframe of either UCDP or PRIO data, with column index of prio grid cells
        for each entry. Data is found at PRIO or UCDP websites XXXX.
    key: str
        Dataframe key for chosen predictor to be extracted and spatially
        represented in the output array.
    prio_key: str
        Dataframe key for the column noting the PRIO grid cell location of each
        entry. For UCDP csvs prio_key = 'priogrid_gid', for PRIO csvs prio_key
        = 'gid'
    debug: bool
        controls debugging print statements

    Returns
    -------
    array:
        array of height 360, width 720 containing the selected parameter arranged
        spatially into the corresponding PRIO grid cell.
    """

    # PRIO grid is 360 * 720
    array = np.zeros(360 * 720)
    prio_grid = dataframe[prio_key]
    for i in range(len(prio_grid)):
        j = prio_grid.iloc[i] - 1
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
    verbose=False,
):
    """Constructs Series of global representations of PRIO and UCDP conflict predictors

    Constructs Series of global representations of PRIO and UCDP conflict
    predictors in an image array format. Image representations of the selected
    conflict predictors at each month between start and stop dates are predicted
    into a global prio grid represenation of size 360 x 720. Each selected predictor
    is allocated an image channel. The output is returned as size: (months,
    predictor number, 360, 720). The output is of size 360 x 720 due to each
    prio grid cell being of dimensions 0.5 degrees lattitude and longitude.

    Parameters
    ----------
    dataframe_prio: pandas DataFrame
        Dataframe constructecd from PRIO grid data CSV in the format supplied
        by the PRIO grid Project.
    dataframe_ucdp: pandas DataFrame
        Dataframe constructecd from UCDP grid data CSV in the format supplied
        by the UCDP grid Project.
    key_list_prio: str, list
        List of Dataframe keys for desired predictors to extract from the PRIO
        Dataframe to represent spatially per prio grid cell.
    key_list_ucdp: str, list
        List of Dataframe keys for desired predictors to extract from the UCDP
        Dataframe to represent spatially per prio grid cell.
    start: int, list
        Representation of data extraction start date. in [y,m,d] format
    stop: int, list
        Representation of data extraction finish date. in [y,m,d] format

    Returns
    -------
        array:
            returns array of size (months, channels, height, width). The array
            is a sequence of global images of Conflict predictors for each month
            between specified start and end dates. Each channel represents a
            specified conflict predictor. Each pixel corresponds to one PRIO grid
            cell.
    """
    # stop  = '2014-01-01'
    num_month = monotonic_date(stop, start)
    comb_channel_len = len(key_list_prio) + len(key_list_ucdp)

    array = np.zeros((num_month, comb_channel_len, 360, 720))
    stop = date_to_int_list(stop)

    if verbose:
        print(num_month)
        print(comb_channel_len)

    month = 0
    extract_month = monotonic_date("2012-01-01")
    for i in range(start[0], stop[0]):
        for j in range(12):  # for each month
            # now fill in selected channels as requried.

            array[month][: len(key_list_ucdp)] = construct_channels(
                dataframe_ucdp[dataframe_ucdp.mon_month == extract_month],
                key_list=key_list_ucdp,
                prio_key="priogrid_gid",
            )
            array[month][len(key_list_ucdp) :] = construct_channels(
                dataframe_prio[dataframe_prio.year == i],
                key_list=key_list_prio,
                prio_key="gid",
            )
            if verbose:
                print(extract_month)

            month += 1
            extract_month += 1
    del month
    return array


def construct_channels(dataframe, key_list, prio_key="gid"):
    """Constructs global parameter layer of multiple specified conflict predictors

    Constructs global parameter layer of multiple specified conflict predictors
    for use in Construct_Combined_sequence. Takes dataframe of either PRIO or
    UCDP conflict data for a single month and selects predictors for channels
    based on the key_list argument.

    Parameters
    ----------
    dataframe: pandas Dataframe
        Dataframe of conflict data, constructed from CSV datasets from PRIO or
        UCDP. Must have a column of PRIO grid cell values for each Predictor
        value.
    key_list: str, list
        List of Dataframe keys for which parameters are extracted into image
        layers.
    prio_key: str
        Dataframe key for the PRIO grid column. For PRIO converted datasets the
        key is 'gid'. For UCDP converted datasets the key is 'priogrid_gid'

    Returns
    -------
    array:
        image representation of global values of conflict data. Returns image of
        dimension (len(key_list, 360, 720). Each pixel represents the predictor
        value at a particular grid cell.
    """
    array = np.zeros((len(key_list), 360, 720))
    for i, keys in enumerate(key_list):
        array[i] = construct_layer(dataframe, key=keys, prio_key=prio_key)
    return array


def date_column(dataframe, baseline=[1989, 1, 1], date_start_key="date_start"):
    """Simple function to add monotonically increasing month to Dataframe

    Adds column entitled 'mon_month' to UCDP dataframes encoding the time ellapsed
    in months between the event start date and a baseline measurng period.

    Parameters
    ----------
    dataframe: pandas Dataframe
        dataframe of ucdp conflict events. Must have a column of start dates in
        format "yyyy-mm-dd" stored as a string. The column key should be equal
        to date start key.
    baseline: int, list
        The start date from which the elapsed time in months is computed from
        i.e month 0. Must be a list of integers in format [y,m,d]
    date_start_key: str
        Dataframe key for the column containing date starts

    """
    # find string dates in dataframe
    vals = dataframe[date_start_key].values
    new_col = np.array([monotonic_date(string_date) for string_date in vals])
    dataframe["mon_month"] = new_col


def h5py_conversion(data_array, filename, key_list_ucdp, key_list_prio):
    # this is for saving the default 360:720 file to extract sequences from
    f = h5py.File("{}.hdf5".format(filename), "w")

    f.create_dataset("data_combined", data=data_array)

    f.close()

    csv = open(filename + "_config.csv", "w")
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

    height = input_data.shape[-2]
    width = input_data.shape[-1]
    # this is not efficient.
    for i in range(height - chunk_size + 1):
        for j in range(width - chunk_size + 1):
            input_data[0][i : i + chunk_size, j : j + chunk_size]

    plt.imshow(input_data)


def random_pixel_bounds(i, j, chunk_size=16):
    """Returns range of indices to extract a randomly placed square of size
    chuksize in which the specified entry of the overall array is captured

    Function returns 4 parameters: i_lower, i_upper, j_lower, j_upper which define
    the bounds of the square region to be extracted. structured so the extracted
    region may be defined as extracted = target[i_lower:i_upper, j_lower:j_upper]

    Parameters
    ----------
    i: int
        i index of the array entry to be captured
    j: int
        j index of the array entry to be captured
    chunk_size: int
        side length of the square to be cut out of the array.

    Returns
    -------
    i_lower: int
        Lower i index for the random square placed over the targetted event.
    i_upper: int
        Upper i index for the random square placed over the targetted event.
    j_lower: int
        Lower j index for the random square placed over the targetted event.
    j_upper: int
        Upper j index for the random square placed over the targetted event.
    """
    # returns the bounds of the image to select with a random pixel size.

    height = random.randint(0, chunk_size - 1)
    width = random.randint(0, chunk_size - 1)
    # this randomly generates a subsample of the image in which the chosen pixel
    # is located randomly in the cut out image.
    i_lower = i - height
    i_upper = i + (chunk_size - height)

    j_lower = j - width
    j_upper = j + (chunk_size - width)

    return [i_lower, i_upper, j_lower, j_upper]


def random_selection(image, i, j, chunk_size=16):
    """Quick method of visualising randomly selected array portion

    Uses random_pixel_bounds to select a portion of an array.

    Parameters
    ----------
    image: array
        Array from which the random selection to be displayed is extracted
    i: int
        i index of array entry which will be included in the random selection
    j: int
        j index of array entry which will be included in the random selection
    chunk_size:
        Edge size of image to be extracted.
    """

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
    debug=False,
):
    """Extracts conflict image sequence samples for a given monnth.

    Produces conflict image sequence samples for each month from an input global
    image sequnce produced using construct_combined_sequence. Takes input image
    sequence of global representations of PRIO grid conflict predictors and extracts
    image sequences of events. Events may be clustered to remove outliers via
    Kmeans clustering. A square of side length chunksize surrounding each non zero
    grid cell in the first channel of each month's prio image is randomly placed
    and the image for that month is extracted as the truth value for prediction.
    The preceding 10 months events are extracted in the same manner as the input.
    If the extracted preceding sequence contains more than min_events non zero
    values in the first image channel (total) the preceding image sequence and
    ground truth prediction are kept.

    Parameters
    ----------
    image: array
        Input array of sequences of global representations of the PRIO conflict
        predictors. This should be produced by the construct_combined_sequence
        function. The array should be of size (months, channels, 360, 720)
    sequence_step: int
        The chosen month from which the prediction ground truth is extracted,
        and which precedes the 10 predicting months, from which the image
        prediction sequence will be extracted.
    chunk_size: int
        side length of the square to be cut out of the array.
    draws: int
        The number of times each event is to be randomly sampled. i.e how many
        data samples are extracted per viable event
    cluster: bool
        Controls whether events will be subject to outlier detection before
        being sampled. Enabling removes small, isolated events.
    min_events: int
        The total number of conflict events required in the prediction sequence
        for a datasample to be accepted. 25 works as an optimum value for filtering
        out small events with little spatial dependancy
    debug: bool
        controls debugging print statements.

    Returns
    -------
    tuple:
        Returns tuple of two numpy arrays. The first contains an array of sampled
        prediction sequences, the second contains an array of ground truth next
        steps in the conflict sequence. The prediction sequence array is of dimensions
        (number of samples, sequence steps, channels, chunksize, chunksize)
        The array of ground truths is of size (number of samples, chunksize,
        chunksize)
    """
    if debug:
        print("Image shape is:", image.shape)

    # image is seq, channels, height, width
    assert sequence_step > 10, (
        "This function selects the datapoints from this test set that contain"
        "a conflict event and then selects predictor data from the 10 preceding steps"
        " as a result i > 10 must be true"
    )
    # locate events occuring in the first channel
    y, x = np.where(image[sequence_step][0] >= 1)

    if cluster:

        X = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))

        clf = LocalOutlierFactor(n_neighbors=5, contamination=0.05)
        y_pred = clf.fit_predict(X)
        bools = y_pred == 1
        non_outs = X[bools]
        x = non_outs[:, 0]
        y = non_outs[:, 1]

        # correct this but answer is this shape

    if debug:
        print(x.shape)

    truth_list = []
    predictor_list = []

    for i, j in zip(y, x):  # now over sites where fatalities have occured
        for _ in range(draws):
            i_lower, i_upper, j_lower, j_upper = random_pixel_bounds(
                i, j, chunk_size=chunk_size
            )

            truth = image[sequence_step][0, i_lower:i_upper, j_lower:j_upper]
            predictors = image[
                sequence_step - 10 : sequence_step, :, i_lower:i_upper, j_lower:j_upper
            ]

            if np.count_nonzero(predictors[:, 0]) >= min_events:
                truth_list.append(truth)
                predictor_list.append(predictors)

    # finally we combine the selected arrays.
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
    """Dummy function to add column of 1s
    to pandas Dataframe"""
    new_col = np.ones(len(dataframe))
    dataframe["binary_event"] = new_col


def nan_to_one(dataframe, key):
    """Replaces NaNs in dataframe column with 0s"""
    dataframe[key] = dataframe[key].fillna(0)


def full_dataset_h5py(
    image,
    filename,
    key_list_prio,
    key_list_ucdp,
    chunk_size=16,
    draws=5,
    min_events=25,
    debug=False,
):
    """Produces h5py Dataset of conflict image sequences, and the prediction ground truth

    Uses random_grid_selection to extract conflict image prediction sequences
    and the next step in the sequence (i.e the image to be predicted). The produced
    image sequences are stored in a hdf5 dataset to allow lazy loading of image
    sequences. The keys 'predictor' and 'truth' are used to store the predictors
    and prediction ground truths. The data selected is stored as meta data attributes
    on the hdf5 dataset.

    Parameters
    ----------
    image: array
        Input array of sequences of global representations of the PRIO conflict
        predictors. This should be produced by the construct_combined_sequence
        function. The array should be of size (months, channels, 360, 720)
    filename: str
        name of hdf5 file to be saved.
    key_list_prio: str, list
        List of PRIO data keys selected
    key_list_ucdp: str, list
        List of UCDP data keys selected
    chunk_size: int
        dimensions of images in image sequence to be extracted.
    draws: int
        Number of times an event is to be randomly sampled
    min_events: int
        Threshold of conflict events in image sequence that the image sequence
        must meet to be added to the dataset.
    debug: bool
        Switch for turning on debugging print options

    """

    with h5py.File(filename + ".hdf5", "w") as f:
        for i in range(11, len(image)):
            print(i)
            t1, t2 = random_grid_selection(image, i, min_events=min_events)
            if i == 11:
                # creat h5py file at first step.
                f.create_dataset(
                    "predictor", data=t1, maxshape=(None, None, None, None, None)
                )  # compression="gzip", chunks=True, taken out
                f.create_dataset("truth", data=t2, maxshape=(None, None, None))

            else:
                # expand the dataset depending on how many samples we extract
                # that meet thresholds. This is a random process so cannot
                # predefine file size.
                f["predictor"].resize(
                    (f["predictor"].shape[0] + t1.shape[0]), axis=0
                )  # expand dataset
                f["truth"].resize((f["truth"].shape[0] + t2.shape[0]), axis=0)

                f["predictor"][
                    -t1.shape[0] :
                ] = t1  # place new data in expanded dataset
                f["truth"][-t2.shape[0] :] = t2

        f["predictor"].attrs.create("key_prio", np.string_(key_list_prio))
        f["truth"].attrs.create("key_ucdp", np.string_(key_list_ucdp))


def data_set_analysis(dataset):
    dat = np.zeros(len(dataset["predictor"]))
    for i in range(len(dataset["predictor"])):
        dat[i] = len(np.where(f["predictor"][i][:, 0] > 0)[0])
        if (i % 1000) == 0:
            print(i)
    return dat


def find_avg_lazy_load(data, div=10000):
    """Extracts average and std from produced hdf5 datasets

    Uses hdf5 lazy loading to subdivide produced image sequence datasets and extract
    an overall average, when the produced datasets are too large to naively average
    and find the standard deviation for.

    Parameters
    ----------
    data: hdf5 file
        Takes loaded hdf5 file. File should have two datasets, accessible via
        the keys: 'predictor' and 'truth'. Full_dataset_h5py produces suitable
        files.
    div: int
        The number of datasamples to average over at a time.

    Returns
    -------
    avg: list of int
        List of averages for each channel in the input image sequence dataset
    std: List of int
        List of standard deviations for each channel in the inout image sequence
        dataset.
    """

    predictor = data["predictor"]
    channel_num = predictor.shape[2]
    avg = np.zeros(channel_num)
    std = np.zeros_like(avg)

    for i in range(channel_num):
        # batching as cant fit into ram

        batch_avg = 0
        batch_std = 0
        for j in range(int(len(predictor) / div)):

            batch_avg += np.sum(predictor[j * div : (j + 1) * div, :, i])

            batch_std += np.sum(
                predictor[j * div : (j + 1) * div, :, i]
                * predictor[j * div : (j + 1) * div, :, i]
            )

        batch_avg += np.sum(
            predictor[int(len(predictor) / div) * div : len(predictor), :, i]
        )
        batch_std += np.sum(
            predictor[int(len(predictor) / div) * div : len(predictor), :, i]
            * predictor[int(len(predictor) / div) * div : len(predictor), :, i]
        )

        sing_chan_shape = np.array(predictor.shape)
        sing_chan_shape[2] = 1
        batch_avg /= np.prod(sing_chan_shape)
        batch_std /= np.prod(sing_chan_shape)

        batch_std = np.sqrt(batch_std - batch_avg ** 2)

        avg[i] = batch_avg
        std[i] = batch_std

    return avg, std


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


def index_return(ind, x_dim, y_dim):

    x_out = ind % x_dim
    y_out = int(ind / x_dim)
    return y_out, x_out


def coord_to_grid(long, lat, x_dim=720, y_dim=360):
    """Returns grid Coordinates for given longitude, lattitude

    Used for converting longitude and lattitude of conflict events into grid cells


    Parameters
    ----------
    long: float
        Degree of longitude for desired location
    lat: float
        Degree of lattitude for desired location
    x_dim: int, optional
        Number of grid cells in x dimension
    y_dim: int, optional
        Number of grid cells in y dimension

    Returns
    -------
    long: int
        Longitudonal cell index of given location
    lat: int
        Lattitudonal cell index of given location
        """
    lat_dummy = np.arange(-90, 90, 0.5)
    long_dummy = np.arange(-180, 180, 0.5)

    round_long = round(long)
    round_lat = round(lat)

    long = np.where(long_dummy == round_long)
    lat = np.where(lat_dummy == round_lat)
    lat = lat[0][0]
    long = long[0][0]
    return long, lat


def round(i):
    """for rounding - always rounding down."""
    j = int(i)
    k = i - j
    if k > 0.5:  # if above i.5 orrigionally
        j += 0.5
    return j


def regional_selection(
    data_ucdp, north=37.32, south=-34.5115, east=51.2752, west=-17.3113
):
    """Select a box of UCDP data from dataframe based on longitude and lattitude
    """
    data_ucdp = data_ucdp[
        (data_ucdp.latitude >= south)
        & (data_ucdp.latitude <= north)
        & (data_ucdp.longitude >= west)
        & (east >= data_ucdp.longitude)
    ]

    return data_ucdp


def construct_dataset(
    filename,
    data_prio,
    data_ucdp,
    key_list_prio,
    key_list_ucdp,
    start=[2012, 1, 1],
    stop="2014,1,1",
):
    """Produces image sequence dataset from input of conflict predictor dataframes

    Short pipeline function for construct_combined_sequence, and full_dataset_h5py.

    Parameters
    ----------
    filename: str
        The filename underwhich the produced hdf5 image sequence dataset is saved
    data_prio: pandas DataFrame
        Dataframe of PRIO grid v2 data
    data_ucdp: pandas Dataframe
        Dataframe of UCDP conflict data
    key_list_prio: list of str
        List of strings passed to dataframe to select chosen conflict predictors
     key_list_ucdp: list of str
        List of strings passed to dataframe to select chosen conflict predictors
    start: list of int
        start date of data to be selected in int [y,m,d] format
    stop: str
        stop date of data to be selected in "yyyy-mm-dd" format.
    """
    test_set = construct_combined_sequence(
        data_prio,
        data_ucdp,
        key_list_prio=key_list_prio,
        key_list_ucdp=key_list_ucdp,
        start=[2012, 1, 1],
        stop="2014-01-01",
    )
    test_set[:, 0][test_set[:, 0] > 0] = 1
    full_dataset_h5py(test_set, filename, key_list_prio, key_list_ucdp)
