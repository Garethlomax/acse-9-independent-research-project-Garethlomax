# Conflict_LSTM
__THE REPORT TO BE ASSESED IS: Masters_project(10).pdf__
## Summary

[![Build Status](https://travis-ci.com/msc-acse/acse-9-independent-research-project-Garethlomax.svg?branch=master)](https://travis-ci.com/msc-acse/acse-9-independent-research-project-Garethlomax)


Current Version: 0.1.0

Lightweight python package implementing easy to use Convolutional LSTMs and Convolutional LSTM encoder - decoder models for use in conflict prediction.

Conflict_LSTM allows:

- The construction and training of spatially deep Convolutional LSTM encoder  decoder models.
- Production of new image sequence datasets from PRIO and UCDP data.
- Analysis and Visualisation of produced results.

./Figures contains generated figures for use in report and presentation

./saved runs contains saved state dict of trained models 

./training logs contains training logs of trained models


### Installation

Download git repository and run **`$ python setup.py install`** inside the directory.

## Requirements

The package dependancies and current requirements including versions are outlined in requirements.txt. These may be installed recursively using pip.

The project is also dependant on Cartopy for use in coordinate transforms while plotting data. Cartopy should be installed using conda, due to its own dependancy on non pip availiable distributions. The plotting functionality that requires Cartopy is isolated to map_module. If users do not wish to use the plotting functionality they may import the other modules. 
__Note that Cartopy is not listed in the requirements.txt if downloading recursively from pip__



## Functionality
Functionality is split across 3 modules: latest_run, hpc_construct, and map_module.

- latest_run contains functionality to construct and train ConvLSTM encoder decoder models to be run on latest conflict prediction data.
- hpc_construct contains functionality to construct new conflict datasets and to analyse predictions.
-map_module contans functionality to visualise conflict data.

## Usage
The package is designed to be use for research in the field of conflict prediction. The functions are designed to be as lightweight and readily customiseable as possible. Example usecases are demonstrated in .ipynb notebooks in the examples folder. 

- dataset_construction_example.ipynb deals with the construction of the dataset.
- model_training_example.ipynb outlines the construction and training of a model.
- analysis_example.ipynb demonstrates the analysis that may be performed on a now constructed model.

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
   """Calculates the Area Under the Reciever Operator Charactersitc (AUROC)
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

