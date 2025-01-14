# -*- coding: utf-8 -*-
"""
author = gareth lomax
"""


# all torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, multilabel_confusion_matrix, roc_curve, roc_auc_score, average_precision_score

import numpy as np
import random

import matplotlib.pyplot as plt
import h5py


import pandas as pd
from hpc_construct import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda'



class LSTMunit(nn.Module):
    """Base unit for an overall convLSTM structure.

    Implementation of ConvLSTM in Pythorch. Used as a worker class for
    LSTMmain. Performs Full ConvLSTM Convolution as introduced in XXXX.
    Automatically uses a padding to ensure output image is of same height
    and width as input.
    Each cell takes an input the data at the current timestep Xt, and a hidden
    representation from the previous timestep Ht-1
    Each cell outputs Ht

    Attributes
    ----------
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


    """
    def __init__(self, input_channel_no, hidden_channels_no, kernel_size, stride = 1, input_dim = 16):
        """Constructor method for LSTM

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
            depractated"""
        super(LSTMunit, self).__init__()
        self.input_channels = input_channel_no

        self.output_channels = hidden_channels_no

        self.kernel_size = kernel_size

        #to ensure output image same output dimensions as input
        self.padding = (int((self.kernel_size - 1) / 2 ), int((self.kernel_size - 1) / 2 ))

        # as in conv nowcasting - see references
        self.stride = stride # for same reasons as above stride must be 1.

        # list of concolution instances for each lstm cell step
        # Filters with Wx_ are unbiased, filters with Wh_ are biased.
        # Stored in module dict to track as parameter.
        self.filter_name_list = ['Wxi', 'Wxf', 'Wxc', 'Wxo','Whi', 'Whf', 'Whc', 'Who']
        self.conv_list = [nn.Conv2d(self.input_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = False).cuda() for i in range(4)]
        self.conv_list = self.conv_list + [(nn.Conv2d(self.output_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = True).cuda()).double() for i in range(4)]
        self.conv_dict = nn.ModuleDict(zip(self.filter_name_list, self.conv_list))

        # of dimensions seq length, hidden layers, height, width
        # statically declaring allows back compatability with previous runs
        shape = [1, self.output_channels, 16, 16]

        # Wco, Wcf, Wci defined as tensors as are multiplicative, not convolutions
        # Tracked as a parameter to allow differentiability.
        self.Wco = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)
        self.Wcf = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)
        self.Wci = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)

        # activation functions.
        self.tanh = torch.tanh
        self.sig  = torch.sigmoid

#     (1, 6, kernel_size=5, padding=2, stride=1).double()
    def forward(self, x, h, c):
        """Pytorch module forward method.

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
        """

        # Calculates as in Nowcasting Paper. see url for explanation
        i_t = self.sig(self.conv_dict['Wxi'](x) + self.conv_dict['Whi'](h) + self.Wci * c)
        f_t = self.sig(self.conv_dict['Wxf'](x) + self.conv_dict['Whf'](h) + self.Wcf * c)
        c_t = f_t * c + i_t * self.tanh(self.conv_dict['Wxc'](x) + self.conv_dict['Whc'](h))
        o_t = self.sig(self.conv_dict['Wxo'](x) + self.conv_dict['Who'](h) + self.Wco * c_t)
        h_t = o_t * self.tanh(c_t)
        return h_t, c_t


class LSTMmain(nn.Module):
    """Full ConvLSTM module

    Full implementation of a ConvLSTM, for use alone or as part of an encoder
    decoder model. The class initialises and iterates over collections of ConvLSTM
    units.

    Attributes
    ----------

    input_channel_no: int
        The number of input channels in the input image sequence
    hidden_channel_no: int
        The number of channels in the output image sequence
    kernel_size: int
        The size of the kernel used in the convolutional opertation.
    hidden_channel_structure: int, list
        Describes the number of hidden channeels in the layers of the multilayer
        ConvLSTM. Is a list of minimum length 1. i.e a ConvLSTM with 3 layers
        with 2 hidden states in each would have hidden_channel_structure = [2,2,2]
    copy_bool : boolean, list
        List of booleans for each layer of the ConvLSTM specifying if a hidden
        state is to be copied in as the initial hidden state of the layer. This
        is for use in encoder - decoder architectures
    debug: boolean
        Controls print statements for debugging

    """


    def __init__(self, shape, input_channel_no, hidden_channel_no, kernel_size, layer_output, hidden_channel_structure, copy_bool = False, debug = False, save_outputs = True, decoder = False, second_debug = False):
        super(LSTMmain, self).__init__()


        self.copy_bool = copy_bool

        self.hidden_channel_structure = hidden_channel_structure

        self.debug = debug
        self.second_debug = second_debug
        self.save_all_outputs = save_outputs

        self.shape = shape

        #number of layers in the encoder.
        self.layers = len(hidden_channel_structure)

        self.seq_length = shape[1]

        self.enc_len = len(shape)

        self.input_chans = input_channel_no

        self.hidden_chans = hidden_channel_no

        self.kernel_size = kernel_size

        self.layer_output = layer_output

        # initialise the different conv cells.
        # allows test input to be an array
        self.full_channel_list = [input_channel_no]
        self.full_channel_list.extend(list(self.hidden_channel_structure))


        # initialises units for each layer in LSTM
        self.unit_list = nn.ModuleList([(LSTMunit(self.full_channel_list[i], self.full_channel_list[i+1], kernel_size).double()).cuda() for i in range(len(self.hidden_channel_structure))])

        if self.debug:
            print("full_channel_list:")
            print(self.full_channel_list)
            print("number of units:")
            print(len(self.unit_list))

    def forward(self, x, copy_in = False, copy_out = [False, False, False]):
        """Forward method of the ConvLSTM

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

        """


        internal_outputs = []

        layer_output = [] # empty list to save each h and c for each step.


        # x is 5th dimensional tensor.
        # x is of size batch, sequence, layers, height, width

        # these need to be of dimensions (batchsizze, hidden_dim, heigh, width)


        # hidden state is of shape [1, layer output channels, height, width]
        h_shape = list(x.shape[:1] + x.shape[2:]) # seq is second, we miss it with fancy indexing
        h_shape[1] = self.hidden_chans
        if self.debug:
            print("h_shape:")
            print(h_shape)


        # stores initial hidden states for each layer.
        empty_start_vectors = []

        k = 0 # to count through our input state list.
        for i in range(self.layers):
            if self.copy_bool[i]:
                # uses supplied hidden state from an encoder
                empty_start_vectors.append(copy_in[k])
                k += 1 # iterate through input list.
            else:
                # for layers without supplied hidden state initialise new hidden
                # state. Called on every forward to avoid computational tree
                # overlap between minibatches.
                assert self.copy_bool[i] == False, "copy_bool arent bools"

                h_shape = list(x.shape[:1] + x.shape[2:]) # seq is second
                h_shape[1] = self.full_channel_list[i+1]
                # append new hidden state and cell memory
                empty_start_vectors.append([(torch.zeros(h_shape).double()).cuda(), (torch.zeros(h_shape).double()).cuda()])

        del k # clear up k so no spare variables flying about.

        if self.debug:
            for i in empty_start_vectors:
                print(i[0].shape)
            print(" \n \n \n")


        # pass input sequence through each layer in the deep LSTM.
        for i in range(self.layers):
            # stores output for each layer in the LSTM to pass as sequence input
            # to next layer
            layer_output = []

            if self.debug:
                print("layer iteration:")
                print(i)

            # copy in initial hidden states and cell memory for each layer.
            # this is used for encoder decoder architectures.
            # default is to put in empty vectors.
            h, c = empty_start_vectors[i]

            if self.debug:
                print("new h shape")
                print(h.shape)

            #iterates over sequence
            for j in range(self.seq_length):

                if self.debug:
                    print("inner loop iteration:")
                    print(j)
                    print("x dtype is:" , x.dtype)
                    print("inner loop size:")
                    print(x[:,j].shape)
                    print("h size:")
                    print(h.shape)

                #call forward method of layer LSTM unit and calcualate next
                # timesteps hidden state.
                h, c = self.unit_list[i](x[:,j], h, c)

                # saves hidden state and cell memory at each timestep to pass
                # to next layer.
                layer_output.append([h, c])


            if copy_out[i] == True:
                # saves last hidden state of each layer to pass to decoder to unroll
                # when used in an encoder decoder format.
                internal_outputs.append(layer_output[-1])

            # extract hidden states to pass as x input to next layer
            h_output = [i[0] for i in layer_output]
            if self.debug:
                print("h_output is of size:")
                print(h_output[0].shape)

            x = torch.stack(h_output,0)
            x = torch.transpose(x, 0, 1)

            if self.debug:
                print("x shape in LSTM main:" , x.shape)
                print("x reshaped dimensions:")
                print(x.shape)

        return x , internal_outputs


class LSTMencdec_onestep(nn.Module):
    """Class to allow easy construction of ConvLSTM Encoder-Decoder models

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

    Attributes
    ----------

    Structure: int, list


    """

    def __init__(self, structure, input_channels, kernel_size = 5, debug = True):
        """Constructor for LSTMencdec

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
        """


        super(LSTMencdec_onestep, self).__init__()
        assert len(structure.shape) == 2, "structure should be a 2d numpy array with two rows"

        self.debug = debug

        # shape of input
        shape = [1,10,3,16,16]

        self.structure = structure
        self.input_channels = input_channels
        self.kernel_size = kernel_size

        # extract arguments for encoder and decoder constructors from structure
        self.enc_shape, self.dec_shape, self.enc_copy_out, self.dec_copy_in = self.input_test()

        if self.debug:
            print("enc_shape, dec_shape, enc_copy_out, dec_copy_in:")
            print(self.enc_shape)
            print(self.dec_shape)
            print(self.enc_copy_out)
            print(self.dec_copy_in)





        #initialise encoder and decoder structures
        self.encoder = LSTMmain(shape, self.input_channels, len(self.enc_shape)+1, self.kernel_size, layer_output = self.enc_copy_out, hidden_channel_structure = self.enc_shape, copy_bool = [False for k in range(len(self.enc_shape))]  ).cuda()

        # now one step in sequence
        shape = [1,1,1,16,16]

        self.decoder = LSTMmain(shape, self.enc_shape[-1], len(self.dec_shape), self.kernel_size, layer_output = 1, hidden_channel_structure = self.dec_shape, copy_bool = self.dec_copy_in,  second_debug = False).cuda()




    def input_test(self):
        """Checks and extracts information from the given structure argument

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
        """
        copy_grid = []
        # finds dimensions of the encoder
        enc_layer = self.structure[0]
        enc_shape = enc_layer[enc_layer!=0]
        dec_layer = self.structure[1]
        dec_shape = dec_layer[dec_layer!=0]

        # find vertical overlaps between non zero elements
        for i in range(len(enc_layer)):
            if self.debug:
                print(enc_layer[i], dec_layer[i])
            if (enc_layer[i] != 0) and (dec_layer[i] != 0):
                copy_grid.append(True)
            else:
                copy_grid.append(False)


        enc_overlap = copy_grid[:len(enc_layer)-1]

        num_dec_zeros = len(dec_layer[dec_layer==0])

        dec_overlap = copy_grid[num_dec_zeros:]

        return enc_shape, dec_shape, enc_overlap, dec_overlap

    def forward(self, x):
        """Forward method of LSTMencdec

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
        # pass inputs to encoder
        x, out_states = self.encoder(x, copy_in = False, copy_out = self.enc_copy_out)

        # select last input of encoder for conditional encoder decoder model.
        x = x[:,-1:,:,:,:]

        res, _ = self.decoder(x, copy_in = out_states, copy_out = [False, False, False,False, False])
        if self.debug:
            print("FINISHING ONE PASS")
        return res



class HDF5Dataset(Dataset):
    """dataset wrapper for hdf5 dataset to allow for lazy loading of data. This
    allows ram to be conserved.

    As the hdf5 dataset is not partitioned into test and validation, the dataset
    takes a shuffled list of indices to allow specification of training and
    validation sets.
    """

    def __init__(self, path, index_map, transform = None):
        """constructor for dataset

        Parameters
        ----------
        path: str
            Path to hdf5 dataset file to load
        index_map: list of ints
            List of shuffled indices. Allows shuffling of hdf5 dataset once extracted.
            The value at the list index is the mapped sample extracted from the
            hdf5 dataset. e.g A index list of [2,1,3] would mean that if the
            2nd value was called via __getitem__ by a dataloader, the 1st value
            in the dataframe would be returned. This provides less overhead than
            shuffling each selection
        """
        self.path = path

        self.index_map = index_map # maps to the index in the validation split
        # due to hdf5 lazy loading index map must be in ascending order.

        self.file = h5py.File(path, 'r')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self,i):

        i = self.index_map[i] # index maps from validation set to select new orders

        if isinstance(i, list):
            # i is only a list when using fancy or random indexing
            # sorts into ascending order as hdf5 may only be queried in ascending
            # order
            i.sort()


        predictor = torch.tensor(self.file["predictor"][i])

        truth = torch.tensor(self.file["truth"][i])

        return predictor, truth

def initialise_dataset_HDF5(valid_frac = 0.1, dataset_length = 9000):
    """
    Returns datasets for training and validation.

    Loads hdf5 custom dataset and utilising a shuffle split, dividing according
    to specified validation fraction.

    Parameters
    ----------
    valid_frac: float
        fraction of the loaded dataset to portion to validation
    dataset_length: int
        number of samples in the dataset to be loaded.

    Returns
    -------
    train_dataset: Pytorch dataset
        Dataset containing shuffled subset of samples for training
    validation_dataset: Pytorch dataset
        Dataset containing shuffled subset of samples for validation

    """

    if valid_frac != 0:

        dummy = np.array(range(dataset_length)) # clean this up - not really needed

        train_index, valid_index = validation_split(dummy, n_splits = 1, valid_fraction = 0.1, random_state = 0)

        train_dataset = HDF5Dataset("train_set.hdf5", index_map = train_index)

        valid_dataset = HDF5Dataset("test_set.hdf5", index_map = valid_index)

        return train_dataset, valid_dataset

    else:
        print("not a valid fraction for validation") # turn this into an assert.

def initialise_dataset_HDF5_full(dataset, valid_frac = 0.1, dataset_length = 9000, avg = 0, std = 0, application_boolean = [0,0,0,0,0]):
    """
    Returns datasets for training and validation.

    Loads hdf5 custom dataset and utilising a shuffle split, dividing according
    to specified validation fraction.

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

    """

    if valid_frac != 0:

        dummy = np.array(range(dataset_length))

        train_index, valid_index = validation_split(dummy, n_splits = 1, valid_fraction = 0.1, random_state = 0)

        train_dataset = HDF5Dataset_with_avgs(dataset,train_index, avg, std, application_boolean)

        valid_dataset = HDF5Dataset_with_avgs(dataset,valid_index, avg, std, application_boolean)


        return train_dataset, valid_dataset

    else:
        print("not a valid fraction for validation") # turn this into an assert.



def validation_split(data, n_splits = 1, valid_fraction = 0.1, random_state = 0):
    """Produces a stratified shuffle split of given dataset

    Wrapper around sklearn functions to produce stratified shuffle split of given
    dataset

    """
    dummy_array = np.zeros(len(data))
    split = StratifiedShuffleSplit(n_splits, test_size = valid_fraction, random_state = 0)
    generator = split.split(torch.tensor(dummy_array), torch.tensor(dummy_array))
    return [(a,b) for a, b in generator][0]

def unsqueeze_data(data):
    """
    Takes in moving MNIST object - must then account for
    """

    # split moving mnist data into predictor and ground truth.
    predictor = data[:][0].unsqueeze(2)
    predictor = predictor.double()

    truth = data[:][1].unsqueeze(2)# this should be the moving mnist sent in
    truth = truth.double()

    return predictor, truth




def train_enc_dec(model, optimizer, dataloader, loss_func = nn.MSELoss(), verbose = False):
    """Training function for encoder decoder models.

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
    """
    i = 0
    model.train()
    # model now tracks gradients
    tot_loss = 0
    for x, y in dataloader:
        x = x.to(device) # Copy image tensors onto GPU(s)
        y = y.to(device)
        optimizer.zero_grad()
        # zeros saved gradients in the optimizer.
        # prevents multiple stacking of gradients

        prediction = model(x)

        if verbose:
            print(prediction.shape)
            print(y.shape)
        loss = loss_func(prediction[:,0,0], y)

        # differentiates model parameters wrt loss
        loss.backward()

        optimizer.step()
        # steps forward model parameters

        tot_loss += loss.item()

        if verbose:
            print("BATCH:")
            print(i)
        i += 1

        if verbose:
            print("MSE_LOSS:", tot_loss / i)
        tot_loss /= i
    return model, tot_loss # trainloss, trainaccuracy

def validate(model, dataloader, loss_func = nn.MSELoss(), verbose = False):

    """as for train_enc_dec but without training - and acting upon validation
    data set
    """
    tot_loss = 0
    i = 0
    model.eval()
    # model no longer tracks gradients.

    for x, y in dataloader:
        with torch.no_grad():
            # no longer have to specify tensor gradients
            # equivalent to as volatile = True for deprecated scripts.

            x=x.to(device)
            y=y.to(device)
            # send to GPU(s)

            prediction = model(x)

            loss = loss_func(prediction[:,0,0], y)

            tot_loss += loss.item()
            i += 1

            if verbose:
                print("MSE_VALIDATION_LOSS:", tot_loss / i)



    return tot_loss / i # returns total loss averaged across the dataset.

def train_main(model, params, train, valid, epochs = 30, batch_size = 1):
    """Main training functions for the LSTMencdec

    Iterates over train_enc_dec to train the given model.

    Parameters
    ----------
    model: pytorch module
        returns the trained model after one epoch, i.e exposure to every piece
        of data in the dataset.
    train
    """
    # make sure model is ported to cuda
    # make sure seed has been specified if testing comparative approaches

#     if model.is_cuda == False:
#         model.to(device)

    # initialise optimizer on model parameters
    # chann
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005, amsgrad= True)
    loss_func = nn.MSELoss()
#     loss_func = nn.BCELoss()
#     loss_func = pytorch_ssim.SSIM()

    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True) # implement moving MNIST data input
    validation_loader = DataLoader(valid, batch_size = batch_size, shuffle = False) # implement moving MNIST

    for epoch in range(epochs):

        train_enc_dec(model, optimizer, train_loader, loss_func = loss_func) # changed


        torch.save(optimizer.state_dict(), F"Adam_new_ams_changed"+str(epoch)+".pth")
        torch.save(model.state_dict(), F"Test_new_ams_changed"+str(epoch)+".pth")


#         validate(model, validation_loader)

    return model, optimizer


class HDF5Dataset_with_avgs(Dataset):
    """dataset wrapper for hdf5 dataset to allow for lazy loading of data. This
    allows ram to be conserved.

    As the hdf5 dataset is not partitioned into test and validation sets, the dataset
    takes a shuffled list of indices to allow specification of training and
    validation sets. The dataet lazy loads from hdf5 datasets and applies standard
    score normalisation in the __getitem__ method.

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

    def __init__(self, path, index_map, avg, std, application_boolean, transform = None):
        """Constructor for hdf5 dataset
        """
        self.path = path

        self.index_map = list(index_map)
        # maps from the requested __getitem__ index to the shuffled index in its
        # place.
        self.avg = avg
        self.std = std
        self.application_boolean = application_boolean
        self.file = h5py.File(path, 'r')

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self,i):

        i = self.index_map[i]
        # index maps to shuffled position
        if isinstance(i, list):
            # if i is a list.
            # sorts into ascending order
            i.sort()

        predictor = torch.tensor(self.file["predictor"][i])



        truth = torch.tensor(self.file["truth"][i])

        #normalise along dimensions depending on fancy index use.
        if isinstance(i, list):
            for j in range(len(self.avg)):
                if self.application_boolean[j]:
                    predictor[:,:,j] -= self.avg[j]
                    predictor[:,:,j] /= self.std[j]

        else:
            for j in range(len(self.avg)):
                if self.application_boolean[j]:
                    predictor[:,j] -= self.avg[j]
                    predictor[:,j] /= self.std[j]

        return predictor, truth



def wrapper_full(name, optimizer,  structure, loss_func, avg, std, application_boolean, lr = None, epochs = 50, kernel_size = 3, batch_size = 50, dataset_name = 'train_fixed_25.hdf5'):
    """Training wrapper for LSTM encoder decoder models.

    Trains supplied model using train_enc_dec fucntions. Logs model hyperparameters
    and trainging and validation losses in csv training log. Saves the model and
    optimiser state dictionaries after each epoch in order to allow for easy
    checkpointing.

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
    """
    f = open(name + ".csv", 'w')
    # open csv file for saving

    # construct model and send to GPU(s)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(LSTMencdec_onestep(structure, 5, kernel_size = kernel_size)).to(device)
    else:
        model = LSTMencdec_onestep(structure, 5, kernel_size = kernel_size).to(device)

    # pass model parameters to optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, amsgrad= True)

    # detail hyperparameters in log file
    f.write("Structure: \n")
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            f.write("{},".format(structure[i,j]))
        f.write("\n") # new line

    f.write("Parameters:\n")
    f.write("optimizer, epochs, learning rate, kernel size \n")

    if lr != None:
        f.write("{},{},{},{},{}\n".format("test", epochs, lr, kernel_size, batch_size))
    else:
        f.write("{},{},{},{},{}\n".format("othertest", epochs, "Default", kernel_size, batch_size))

    f.write("loss_func:\n")
    f.write(loss_func.__repr__() + "\n")

    f.write("optimizer:\n")
    f.write(optimizer.__repr__() + "\n")

    f.write("\n\n\n")
    f.write("TRAINING\n")

    f.close()
    # initialise training and validation datasets.
    train, valid = initialise_dataset_HDF5_full(dataset_name, valid_frac = 0.1, dataset_length = 56413,avg = avg, std = std, application_boolean=application_boolean)

    loss_func = loss_func

    # pass datasets to dataloaders
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True) # implement moving MNIST data input
    valid_loader = DataLoader(valid, batch_size = 2000, shuffle = False) # implement moving MNIST

    for epoch in range(epochs):

        # train the model
        _, loss = train_enc_dec(model, optimizer, train_loader, loss_func = loss_func) # changed

        # save for checkpointing
        torch.save(optimizer.state_dict(), name+str(epoch)+"optimizer.pth")
        torch.save(model.state_dict(), name+str(epoch)+".pth")

        #compute validation at each epoch
        valid_loss = validate(model, valid_loader, loss_func = loss_func) # validation - need to shuffle split.


        f = open(name + ".csv", 'a')
        f.write(str(loss) + "," + str(valid_loss) + "\n")
        f.close()
    return True



def test_image_save(model, train_loader, name, sample = 7):
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
    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break


    with torch.no_grad():
        x = model(a.cuda())

    x = x.cpu()
    # return our prediction from GPU to CPU to be accessed by matplotlib

    fig, axes = plt.subplots(1,2)
    print(x.shape)
    print(b.shape)
    axes[0].imshow(x[sample][0][0])
    axes[1].imshow(b[sample])

    axes[1].set_title("truth")
    axes[0].set_title("Prediction")
    fig.suptitle("Prediction of:" + name)
    fig.savefig(name + "comparison.pdf")
    fig, axes = plt.subplots(10,1,figsize=(32,32))
#    for i in range(10):
#        axes[i].imshow(a[sample][i][0])
    return fig

def f1(model, train_loader, avg = 'macro'):
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
    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break


    with torch.no_grad():
        x = model(a.cuda())


    x = x.cpu()
    x[x>0] = 1
    x[0>x] = 0
    scores = []
    for i in range(len(b)):
        score = f1_score(b[i].view(256).numpy(), x[i,0,0].view(256).numpy(), average=avg)
        scores.append(score)
        truth = set(list(b[i].view(256).numpy()))
        pred = set(list(x[i,0,0].view(256).numpy()))
        if len(truth - pred) > 0:
            print("sample" + i + " is a poor prediction and should be investigated")
    return scores






def metrics(model, train_loader, name = 'default', verbose = True, save = True):
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
    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break

    with torch.no_grad():
        x = model(a.cuda())

    x = x.cpu()
    x[x>0] = 1
    x[0>x] = 0

    # reshape
    truth = b.view(-1,256).numpy()
    pred = x[:,0,0].view(-1,256).numpy()
    tn = 0
    tp = 0
    fn = 0
    fp = 0

    if verbose:
        print(truth.shape)
        print(pred.shape)
    for i in range(len(b)):
        for j in range(256):
            # true positive
            if (truth[i][j] == 1) and (pred[i][j] == 1):
                tp += 1
            # true negative
            if (truth[i][j] == 0) and (pred[i][j] == 0):
                tn += 1

            #false positive
            if (truth[i][j] == 0) and (pred[i][j] == 1):
                fp +=1
            #false negative
            if (truth[i][j] == 1) and (pred[i][j] == 0):
                fn += 1

    prec = tp / (tp + fp)
    rec = tp/ (tp + fn)

    f_1 = 2 * prec * rec / (prec + rec)

    if verbose:
        print("tn:" ,tn)
        print("tp:" , tp)
        print("fn:" , fn)
        print("fp" ,fp)
        print("prec:" ,prec)
        print("rec:" , rec)
        print("f1: ", f_1)

    if save:
        f = open(name + "metrics.csv", 'w')
        for i in [tn, tp, fn, fp, prec, rec, f_1]:

            f.write(str(i) + "\n")

        f.close()
    return [tn, fn, tp, fp, prec, rec, f_1]

def area_under_curve_metrics(model, train_loader, name = 'default', verbose = True, save = True):
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

    sig = nn.Sigmoid()
    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break # train loader cannot be indexed

    with torch.no_grad():
        x = model(a.cuda())

    x = x.cpu()
    truth = b.view(-1,256).numpy()
    pred = sig(x[:,0,0].view(-1,256)).numpy()

    r = roc_auc_score(truth, pred)
    av = average_precision_score(truth, pred)
    if verbose:
        print("AUROC:" + str(r))
        print("AUPR:" + str(av))

    if save:
        f = open(name + "metrics.csv", 'a')
        for i in [r,av]:

            f.write(str(i) + "\n")

        f.close()
    return [r, av]

def brier_score(model, train_loader, name = 'default', verbose = True, save = True):
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
    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break # train loader cannot be indexed


    with torch.no_grad():
        x = model(a.cuda())


    x = x.cpu()
    x[x>0] = 1
    x[0>x] = 0

    diff = (x[:,0,0] - b)**2
    brier = np.average(diff)

    if verbose:
        print("brier score:" + str(brier))

    if save:
        f = open(name + "metrics.csv", 'a')
        f.write(str(brier) + "\n")
        f.close()
    return brier

def full_metrics(model, train_loader, name = 'default'):
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
    metrics(model, train_loader, name = name)
    area_under_curve_metrics(model, train_loader, name = name)
    brier_score(model, train_loader, name = name)

def curves(model, train_loader):
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
    sig = nn.Sigmoid()
    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break

    with torch.no_grad():
        x = model(a.cuda())

    x = x.cpu()
    truth = b.view(-1,256).numpy()
    pred = sig(x[:,0,0].view(-1,256)).numpy()
    plt.figure()
    #over diagonal pixels
    for j in range(16):
        t = b[:,j,j].contiguous().view(-1).numpy()
        p = sig(x[:,0,0,j,j].contiguous().view(-1)).numpy()
        fpr, tpr, thresholds = roc_curve(t, p)
        plt.plot(fpr, tpr)
    plt.plot(fpr, fpr)
    plt.xlim(0, 1.1)
    plt.ylim(0,1.1)

    plt.show()

def batch_loss_histogram(model, train_loader, loss_func):

    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break


    with torch.no_grad():
        x = model(a.cuda())


        x = x.cpu()

        loss = []
        for i in range(len(x)):
            loss.append(loss_func(x[i,:,0],b[i:i+1]).item())





    return loss


def analytics(structure, kernel_size, model_path, dataset_path = 'emerg_25.hdf5', dataset_length = 4940, avg_path = "fixed_25_avg.npy", std_path = "fixed_25_std.npy", sample = 1967):
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
    device = 'cuda'
    test_model = nn.DataParallel(LSTMencdec_onestep(structure, 5, kernel_size = kernel_size, debug = False)).to(device) # added data parrallel
    test_model.load_state_dict(torch.load(model_path + ".pth"))
    test_model.eval()

    avg = np.load(avg_path)
    std = np.load(std_path)
    apbln = [0,1,0,0,1]
    train, valid = initialise_dataset_HDF5_full(dataset_path, valid_frac = 0.1, dataset_length = dataset_length ,avg = avg, std = std, application_boolean=apbln)
    train_loader = DataLoader(train, batch_size = 2000, shuffle = False)

    test_image_save(test_model, train_loader, model_path + "comparison", sample = sample)
    full_metrics(test_model, train_loader, model_path)
    return True
