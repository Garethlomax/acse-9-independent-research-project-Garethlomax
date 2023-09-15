# Conflict_LSTM
## Summary
[![Build Status](https://travis-ci.com/msc-acse/acse-9-independent-research-project-Garethlomax.svg?branch=master)](https://travis-ci.com/msc-acse/acse-9-independent-research-project-Garethlomax)

This is a repository for work relating to experimental work using Convolutional LSTM encoder-decoder networks to model diffusion of UCDP conflict events. Please note that this is a repository for experimental work and as such is quite messy. There are no current plans to share this work until a sufficient amount of time has passed to allow for 4 years of out of sample testing

Current Version: 0.1.0

Conflict_LSTM allows:

- The construction and training of spatially deep Convolutional LSTM encoder  decoder models.
- Production of new image sequence datasets from PRIO and UCDP data.
- Analysis and Visualisation of produced results.

./Figures contains generated figures for use in report and presentation

./saved runs contains saved state dict of trained models 

./training logs contains training logs of trained models

## Contents
This repo contains two main aspects: 
1. The Conv_LSTM python package for implementing Conv_LSTMs in PyTorch - found in ./conflict_lstm
2. Scripts derived from the conflict_lstm package for training models on HPC. These are contained outside of the package due to issues with package installation on the main HPC cluster used for model training.

## Installation

Download git repository and run **`$ python setup.py install`** inside the directory.

## Requirements

The package dependancies and current requirements including versions are outlined in requirements.txt. These may be installed recursively using pip.

The project is also dependant on Cartopy for use in coordinate transforms while plotting data. Cartopy should be installed using conda, due to its own dependancy on non pip availiable distributions. The plotting functionality that requires Cartopy is isolated to map_module. If users do not wish to use the plotting functionality they may import the other modules. These issues are due to dependencies on GDAL, which plays well or poorly depending on your local installation. Google collab is notoriously unreliable in this regard. 
__Note that Cartopy is not listed in the requirements.txt if downloading recursively from pip__

### Functionality
Functionality is split across 3 modules: latest_run, hpc_construct, and map_module.

- latest_run contains functionality to construct and train ConvLSTM encoder decoder models to be run on latest conflict prediction data.
- hpc_construct contains functionality to construct new conflict datasets and to analyse predictions.
-map_module contans functionality to visualise conflict data.

### Usage
The package is designed to be use for research in the field of conflict prediction. The functions are designed to be as lightweight and readily customiseable as possible. Example usecases are demonstrated in .ipynb notebooks in the examples folder. 

- dataset_construction_example.ipynb deals with the construction of the dataset.
- model_training_example.ipynb outlines the construction and training of a model.
- analysis_example.ipynb demonstrates the analysis that may be performed on a now constructed model.

# Getting Started

## How to get it to run (training on PBS cluster. Requirements: 96GB ram, 4 GPUs) 
1. Install requirements for your specific anaconda / virtualenv environment
2. Copy repo into local environment of cluster.
3. Navigate to ./HPC_runs
4. Alter example PBS files in ./HPC_runs to your specific directory paths ect
5. Alter example submission .py files to your requirements (see below) e.g bce_w_4.py. These are found in ./HPC_runs
6. Submit a PBS file.
7. Use notebooks to visualise results, debug training and calculate metrics.

## Modifying training and data analysis
All model training is controlled by the wrapper_full() function. All functionality can be called by running the wrapper function. See Example_model_training.ipynb for an example. Note that in theory this can work on a Jupyter Notebook, but for any meaningful speed you should run with multiple GPUs on a cluster.

(Very) Basic examples of dataset construction and analysis can be found in:
- Example_analytics.ipynb
- Example_model_training.ipynb
- Example_dataset_construction.ipynb

  


# Documentation
## Classes

### LSTMunit

- Base unit for an overall convLSTM structure.

- Implementation of ConvLSTM in Pythorch. Used as a worker class for LSTMmain. Performs Full ConvLSTM Convolution as introduced in XXXX. Automatically uses a padding to ensure output image is of same height and width as input. Each cell takes an input the data at the current timestep Xt, and a hidden representation from the previous timestep Ht-1. Each cell outputs Ht

#### Attributes
    
    input_channels: int
        The number of channels in the input image tensor.
    output_channels: int
        The number of channels in the output image tensor following
        convoluton.
    kernel_size: int
        The size of the kernel used in the convolutional opertation.
    padding: int
        The padding in each of the convolutional operations. Automatically
        calculated so each convolution maintains input image dimensions.
    stride: int
        The stride of input convolutions.
    filter_name_list: str
        List of identifying filter names as used in equations from XXXX.
    conv_dict: dict
        nn.Module Dictionary of pytorch convolution modules, with parameters
        specified by attributes listed above. Stored in Module Dict to make
        accessible to pytorch autograd. See pytorch for explanation of
        computational tree tracking in pytorch.
    shape: int, list
        List of dimensions of image input.
    Wco: double, tensor
        Pytorch parameter tracked tensor for use in hammard operation in
        LSTM logic gates. Is a pytorch parameter to allow computational
        tree tracking.
    Wcf: double, tensor
        Pytorch parameter tracked tensor for use in hammard operation in
        LSTM logic gates. Is a pytorch parameter to allow computational
        tree tracking.
    Wci: double, tensor
        Pytorch parameter tracked tensor for use in hammard operation in
        LSTM logic gates. Is a pytorch parameter to allow computational
        tree tracking.
    tanh: class
        Pytorch tanh class.
    sig: class
        Pytorch sigmoid class.
        
#### Methods
- __init__ : Constructor method for LSTM

        Parameters
        ----------
        input_channel_no: int
            Number of channels of the input image in the LSTM unit
        hidden_channel_no: int
            The number of hidden channels of the image output by the unit
        kernel_size: int
            The dimension of the square convolutional kernel used in the forward
            method
        stride: int
            depractated
            
- forward : Pytorch module forward method.

        Calculates a forward pass of the LSTMunit. Takes in the sequence input
        at a timestep, and previous hidden states and cell memories and returns
        the new hidden state and cell memory, as according to the outline in
        XXXX.

        Parameters
        ----------

        x: tensor, double
            Pytorch tensor of dimensions shape, as specified in class constructor
            tensor should be 3 dimensional tensor of dimensions (input channels,
            height, width). x is the image at a single step of an image sequence
        h: tensor, double
            Pytorch tensor of dimensions (output channels, height, width). h is
            the output hidden state from the last step in the LSTM sequence
        c: tensor, double
            Pytorch tensor of dimensions (output channels, height, width). h is
            the output cell memory state from the last step in the LSTM sequence

        Returns
        -------

        h_t: tensor, double
            Tensor of the new hidden state for the current timestep, Pytorch
            tensor of dimensions (output channels, height, width)
        c_t: tensor, double
            Tensor of the new cell memory state for the current tinestep, Pytorch
            tensor of dimensions (output channels, height, width)
### LSTMmain


- Implementation of ConvLSTM for use both standalone and as part of Encoder Decoder Models

- Instances and iterates over LSTMunits

#### Attributes
    input_channel_no: int
        The number of input channels in the input image sequence
    hidden_channel_no: int
        The number of channels in the output image sequence
    kernel_size: int
        The size of the kernel used in the convolutional opertation.
    test_input: int, list
        Describes the number of hidden channeels in the layers of the multilayer
        ConvLSTM. Is a list of minimum length 1. i.e a ConvLSTM with 3 layers
        with 2 hidden states in each would have test_input = [2,2,2]
    copy_bool : boolean, list
        List of booleans for each layer of the ConvLSTM specifying if a hidden
        state is to be copied in as the initial hidden state of the layer. This
        is for use in encoder - decoder architectures
    debug: boolean
        Controls print statements for debugging

#### Methods

- __init__ : constructor method

- Forward: Forward method of the ConvLSTM

        Takes a sequence of image tensors and returns the hidden state output
        of the final LSTM layer. Takes in hidden state tensors for intermediate
        layers to allow for use in decoder models. The method can copy out specified
        hidden states to allow for use in encoder models.

        Parameters
        ----------
        x : double, tensor
            Input image sequence tensor, of dimensions (minibatch size, sequence
            length, channels, height, width)
        copy_in: list of double tensors
            List of hidden state tensors to be copied in to LSTM layers specified
            by copy_bool. copy_in should only contain the hidden state tensors
            for the require a copied in state. The states should be arranged in
            order, so that the hidden state to be copied into the first layer
            is first.
        copy_out: boolean, list
            List of booleans specifying which layer hidden states are to be copied
            out due to being required to be passed to a decoder.

        Returns
        -------
        x: double, tensor
            Tensor of the hidden state outputs of the final LSTM layer. x is of
            shape (minibatch size, image sequence length, final hidden channel
            number, height, width)
        internal_outputs: tensor, list
            Last hidden states of layers specified by copy_out. To be used in
            encoder LSTM to be copied into decoder LSTM as the copy_in parameter
            
### LSTMencdec_onestep

- Class to allow easy construction of ConvLSTM Encoder-Decoder models

    Constructs ConvLSTM encoder - decoder models using LSTMmain. Takes structure
    argument to specify architecture of initialised model. The structure is a
    2D numpy array. The top row of the array defines the encoder, the bottom
    row defines the encoder. Non zero values in the encoder and decoder rows
    define the number of layers and the hidden channel number for each layer
    of the encoder decoder model. 0 values after a positive in an encoder row
    denote the end of the encoder. 0 values precede the hidden channel
    specification for the decoder. A column overlap between two non zero values
    means that the hidden states are copied out of the corresponding encoder layer
    and into the decoder model. An encoder that has hidden channels of size
    6, and 12, and decoder that reduces the prediction to 6 channels symmetrically
    would be input as: structure = [[6,12,0,],
                                    [0,12,6]]

#### Attributes
    
    Structure: int, list


    """
#### Methods

- __init__: Constructor for LSTMencdec

        Constructs two intances of LSTMmain, one encoder and one decoder. passes
        structure argument to input_test function to produce the analytics of
        the functions.

        Parameters
        ----------
        structure: array of ints
            2d array of ints used to specify structure of encoder decoder. The
            top row of the array defines the encoder, the second row of the array
            defines the decoder. Non zero digits signify then number of channels
            in the hidden state for each layer. Zero digits specify the end of
            the encoder, and are used before the initial digits of the decoder
            as a placeholder. Vertical overlap of non zero digits specifies
            that the hidden state of the encoder layer will be copied as the initial
            state into the corresponding decoder layer. An example outputting
            an image sequence of 5 hidden layers is shown.
            structure = [[6,12,24,0,0,0],
                         [0,0,24,12,8,5]].
        input_channels: int
            The number of channels of each image in the image input sequence to
            the encoder decoder.
        kernel_size: int, optional
            The size of the convolution kernel to be used in the encoder and
            decoder layers.
        debug: bool, optional
            Switch to turn off debugging print statements.
            
            
- input_test : Checks and extracts information from the given structure argument

        Returns
        -------
        enc_shape: list of int
            shape argument specifying hidden layers of the encoder to be passed
            to LSTMmain constructor.
        dec_shape: list of int
            enc_shape: list of int
            shape argument specifying hidden layers of the decoder to be passed
            to LSTMmain constructor.
        enc_overlap: list of bool
            List of boolean values denoting whether each layer of the encoder
            overlaps with a decoder layer in the structure input and thus should
            copy out. To be passed to the LSTMmain 'copy_bool' argument
        dec_overlap: list of bool
            List of boolean values denoting whether each layer of the decoder
            overlaps with an encoder layer in the input and thus should
            copy in a hidden layer. To be passed to the LSTMmain 'copy_bool'
            argument.
            
- Forward : Forward method of LSTMencdec

        Takes input image sequence produces a prediction of the next image in
        the sequence frame using a conditional LSTM encoder decoder structure

        Parameters
        ----------
        x: tensor of doubles
            Pytorch tensor of input image sequences. should be of size (minibatch
            size, sequence length, channels, height, width)

        Returns
        -------
        tensor:
            Tensor image prediction of size (minibatch size, sequence length,
            channels, height, width)
        """
### HDF5Dataset_with_avgs
-     dataset wrapper for hdf5 dataset to allow for lazy loading of data. This allows ram to be conserved.
As the hdf5 dataset is not partitioned into test and validation sets, the dataset
    takes a shuffled list of indices to allow specification of training and
    validation sets. The dataet lazy loads from hdf5 datasets and applies standard
    score normalisation in the __getitem__ method.
    
### Methods

- __init__

    

    Parameters
    ----------
    path: str
        filepath to hdf5 dataset to be loaded.
    index_map: list of ints
        List of shuffled indices. Allows shuffling of hdf5 dataset once extracted.
        The value at the list index is the mapped sample extracted from the
        hdf5 dataset. e.g A index list of [2,1,3] would mean that if the
        2nd value was called via __getitem__ by a dataloader, the 1st value
        in the dataframe would be returned. This provides less overhead than
        shuffling each selection.
    avg: list of floats
        List of averages for every channel in image sequence loaded. Length should
        equal number of channels in dataset image sequence
    std: list of floats
        List of standard deviations for every channel in image sequence loaded.
        Length should equal number of channels in dataset image sequence.
    application_boolean: list of bools
        List of bools indicating whether standard score normalisation is to be
        applied to each layer. Length should equal number of channels in dataset
        image sequence.
        
    """



## Functions 
### Initialise_dataset_HDF5_full

-Returns datasets for training and validation.

-Loads hdf5 custom dataset and utilising a shuffle split, dividing according to specified validation fraction.

    Parameters
    ----------
    dataset: str
        filename / path to hdf5 file.
    valid_frac: float
        fraction of the loaded dataset to portion to validation
    dataset_length: int
        number of samples in the dataset to be loaded.
    avg: list of floats
        Averages for each predictor channel in the input image sequences
    std: list of floats
        Standard deviation for each predictor channel in the input image sequence
    application_boolean: list of bools
        List of booleans specifying if which predictor channels should be standard
        score normalised

    Returns
    -------
    train_dataset: Pytorch dataset
        Dataset containing shuffled subset of samples for training
    validation_dataset: Pytorch dataset
        Dataset containing shuffled subset of samples for validation
        
### train_enc_dec

-Training function for encoder decoder models.

    Parameters
    ----------
    model: pytorch module
        Input model to be trained. Model should be end to end differentiable,
        and be inherited from nn.Module. model should be sent to the GPU prior
        to training, using model.cuda() or model.to(device)
    optimizer: pytorch optimizer.
        Pytorch optimizer to step model function. Adam / AMSGrad is recommended
    dataloader: pytorch dataloader
        Pytorch dataloader initialised with hdf5 averaged datasets
    loss_func: pytorch loss function
        Pytorch loss function
    verbose: bool
        Controls progress printing during training.

    Returns
    -------
    model: pytorch module
        returns the trained model after one epoch, i.e exposure to every piece
        of data in the dataset.
    tot_loss: float
        Average loss per sample for the training epoch
        
        

### wrapper_full

-Training wrapper for LSTM encoder decoder models.

-Trains supplied model using train_enc_dec fucntions. Logs model hyperparameters and trainging and validation losses in csv training log. Saves the model and optimiser state dictionaries after each epoch in order to allow for easy checkpointing.

    Parameters
    ----------
    name: str
        filename to save CSV training logs as.
    optimizer: pytorch optimizer
        The desired optimizer needed to train the model
    structure: array of ints
        Structure argument to be passed to lstmencdec. See LSTMencdec for explanation
        of structure format.
    loss_func: pytorch module
        Loss function to be used to calculate training and validation losses.
        The loss should be a CLASS instance of the pytorch loss function, not
        a functinal implementation.
    avg: list of floats
        List of averages for every channel in image sequence loaded. Length should
        equal number of channels in dataset image sequence
    std: list of floats
        List of standard deviations for every channel in image sequence loaded.
        Length should equal number of channels in dataset image sequence.
    application_boolean: list of bools
        List of bools indicating whether standard score normalisation is to be
        applied to each layer. Length should equal number of channels in dataset
        image sequence.
    lr: float
        Learning rate for the optimizer
    epochs: int
        Number of epochs to train the model for
    kernel_size: int
        Size of convolution kernel for the LSTMencoderdecoder
    batch_size: int
        Number of samples in each training minibatch.

    Returns
    -------
    bool:
        indicates if training has been completed.
    
 ### test_image_save
 """Saves comparison between prediction of the given model and ground truth.

    Parameters
    ----------
    model:
        Trained model to visualise prediction of.
    train_loader:
        Dataloader of dataset from which input sequence to be visualised is stored
    name: str
        Filename to save visualised comparison in.
    sample: int
        Sample of the input dataset to be predicted.

    Returns
    -------
    fig:
        Matplotlib figure to be manipulated outside of program.
    """
### f1 
"""Produces average F1 score for each image prediction in the dataset.

    Parameters
    ----------
    model:
        Trained model to visualise prediction of.
    train_loader:
        Dataloader of dataset from which input sequence to be visualised is stored
    avg: str
        method of averaging over each multilabel image. see sklearn.metrics.f1_score
        for specification of the average key types

    Returns
    -------
    list:
        List of scores for each image sequence prediction
    """
### Metrics
"""Calculate TN, FN, TP, FP, precision, recall and f1 score.

    Calculates the true negative, false negative, true positive, false positive,
    precison, recall, and multilabel f_1 score for the model predictions of a
    supplied data sample. The F1 score is calculated according to XXXX. To offset
    bias in multilabel sampling. Produces CSV storing performance metrics for
    later analysis.

    Parameters
    ----------
    model:
        Trained model to visualise prediction of.
    train_loader:
        Dataloader of dataset in which input sequence to be visualised is stored.
    name: str
        filename to store metrics CSV under.
    verbose: bool
        Controls print output of metrics
    save: bool
        controls whether metrics will be saved in csv

    Returns
    -------
    list:
        true negative, false negative, true positive, false positive,
        precison, recall, and multilabel f_1 score for the model predictions.
    """
### Area_under_curve_metrics
    Calculates the Area Under the Reciever Operator Charactersitc (AUROC)
    curve and the Area Under the Precision Recall (AUPR) curve. Uses
    average_precision_score to calculate AUPR.

    Parameters
    ----------
    model:
        Trained model to visualise prediction of.
    train_loader:
        Dataloader of dataset in which input sequence to be visualised is stored.
    name: str
        filename to store metrics CSV under.
    verbose: bool
        Controls print output of metrics
    save: bool
        controls whether metrics will be saved in csv

    Returns
    -------
    list:
        list of [AUROC, AUPR]


    """
### brier_score
"""Calculates the average brier score for each prediction. Saves prediction
    in metrics csv.

    Parameters
    ----------
    model:
        Trained model to visualise prediction of.
    train_loader:
        Dataloader of dataset in which input sequence to be visualised is stored.
    name: str
        filename to store metrics CSV under.
    verbose: bool
        Controls print output of metrics
    save: bool
        controls whether metrics will be saved in csv

    Returns
    -------
    float:
        brier score.

    """
### full_metrics
"""extracts all performance metrics from one data sample.

    Parameters
    ----------
    model:
        Trained model to visualise prediction of.
    train_loader:
        Dataloader of dataset in which input sequence to be visualised is stored.
    name: str
        filename to store metrics CSV under.
    """
### curves
"""PLots ROC curves for diagonal pixels in image prediction

    Plots ROC curves for diagonal pixels in the image predictions.This is done
    to reduce overcrowding of plots.

    Parameters
    ----------
    model:
        Trained model to visualise prediction of.
    train_loader:
        Dataloader of dataset in which input sequence to be visualised is stored.


    """
### analytics
    """Loads given state dict and extracts metrics.

    Uses test_image save and full_metrics to compile metric report in csv and
    to visualise prediction, from loaded pretrained model statedict. NOTE THAT
    STRUCTURE AND KERNEL SIZE SHOULD BE THE SAME AS THE TRAINING MODEL.

    Parameters
    ----------
    structure: array of int
        Structure to initialise the LSTMencdec_onestep model. See LSTMencdec_onestep
        for explanation.
    kernel_size: int
        Size of the convolutional kernel used in the model
    model_path: str
        relative path to saved model state dict
    dataset_path: str
        relative path to dataset to test on
    dataset_length: int
        Number of samples in dataset.
    avg_path: str
        relative path to dataset averages
    std_path: str
        relative path to dataset standard deviations
    sample: int
        the specific dataset sample to be visualised. Note this does not effect
        how the performance metrics are calculated, which are extracted from all
        samples

    Returns
    -------
    bool:
        True
    """
    
### construct_layer
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
### construct_combined_sequence
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
### construct_channels
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
### random_pixel_bounds
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
### random_grid_selection
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
### full_dataset_h5py
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
    ### find_avg_lazy_load
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
### construct_dataset
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
    

