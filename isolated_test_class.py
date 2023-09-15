# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 02:44:45 2019

@author: Gareth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import random

import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import h5py


class LSTMunit_t(nn.Module):
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

    def __init__(self, input_channel_no, hidden_channels_no, kernel_size, stride=1):
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
        super(LSTMunit_t, self).__init__()

        self.input_channels = input_channel_no

        self.output_channels = hidden_channels_no

        self.kernel_size = kernel_size

        self.padding = (
            int((self.kernel_size - 1) / 2),
            int((self.kernel_size - 1) / 2),
        )  # to ensure output image same dims as input
        # as in conv nowcasting - see references
        self.stride = stride  # for same reasons as above stride must be 1.
        """TODO: CHANGE THIS LAYOUT OF CONVOLUTIONAL LAYERS. """
        """ TODO : DEAL WITH BIAS HERE. """
        """ TODO: CAN INCLUDE BIAS IN ONE OF THE CONVOLUTIONS BUT NOT ALL OF THEM - OR COULD INCLUDE IN ALL? """
        """ TODO: decide whether this should be put into function. """
        """TODO: put correct dimensions of tensor in shape"""
        """TODO: DEFINE THESE SYMBOLS. """
        """TODO: PUT THIS IN CONSTRUCTOR."""
        self.filter_name_list = ["Wxi", "Wxf", "Wxc", "Wxo", "Whi", "Whf", "Whc", "Who"]
        # list of concolution instances for each lstm cell step
        # Filters with Wx_ are unbiased, filters with Wh_ are biased.
        # Stored in module dict to track as parameter.
        self.conv_list = [
            nn.Conv2d(
                self.input_channels,
                self.output_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False,
            )
            for i in range(4)
        ]
        self.conv_list = self.conv_list + [
            nn.Conv2d(
                self.output_channels,
                self.output_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=True,
            ).double()
            for i in range(4)
        ]
        self.conv_dict = nn.ModuleDict(zip(self.filter_name_list, self.conv_list))

        #         self.conv_list = [nn.Conv2d(self.input_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = False) for i in range(4)]
        #         self.conv_list = self.conv_list + [(nn.Conv2d(self.output_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = True)).double() for i in range(4)]
        #         self.conv_list = nn.ModuleList(self.conv_list)

        # of dimensions seq length, hidden layers, height, width
        shape = [1, self.output_channels, 16, 16]

        # Wco, Wcf, Wci defined as tensors as are multiplicative, not convolutions
        # Tracked as a parameter to allow differentiability.
        self.Wco = nn.Parameter((torch.zeros(shape).double()), requires_grad=True)
        self.Wcf = nn.Parameter((torch.zeros(shape).double()), requires_grad=True)
        self.Wci = nn.Parameter((torch.zeros(shape).double()), requires_grad=True)

        #         self.Wco = nn.Parameter((torch.zeros(shape).double()), requires_grad = True)
        #         self.Wcf = nn.Parameter((torch.zeros(shape).double()), requires_grad = True)
        #         self.Wci = nn.Parameter((torch.zeros(shape).double()), requires_grad = True)
        #         self.Wco.name = "test"
        #         self.Wco = torch.zeros(shape, requires_grad = True).double()
        #         self.Wcf = torch.zeros(shape, requires_grad = True).double()
        #         self.Wci = torch.zeros(shape, requires_grad = True).double()

        # activation functions.
        self.tanh = torch.tanh
        self.sig = torch.sigmoid

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
        i_t = self.sig(
            self.conv_dict["Wxi"](x) + self.conv_dict["Whi"](h) + self.Wci * c
        )
        f_t = self.sig(
            self.conv_dict["Wxf"](x) + self.conv_dict["Whf"](h) + self.Wcf * c
        )
        c_t = f_t * c + i_t * self.tanh(
            self.conv_dict["Wxc"](x) + self.conv_dict["Whc"](h)
        )
        o_t = self.sig(
            self.conv_dict["Wxo"](x) + self.conv_dict["Who"](h) + self.Wco * c_t
        )
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t


class LSTMmain_t(nn.Module):
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



    """

    """ collection of units to form encoder/ decoder branches - decide which are which
    need funcitonality to copy in and copy out outputs.


    layer output is array of booleans selectively outputing for each layer i.e
    for three layer can have output on second and third but not first with
    layer_output = [0,1,1]"""

    """TODO: DECIDE ON OUTPUT OF HIDDEN CHANNEL LIST """

    def __init__(
        self,
        shape,
        input_channel_no,
        hidden_channel_no,
        kernel_size,
        layer_output,
        test_input,
        copy_bool=False,
        debug=False,
        save_outputs=True,
        decoder=False,
        second_debug=False,
    ):
        super(LSTMmain_t, self).__init__()

        """TODO: USE THIS AS BASIS FOR ENCODER DECODER."""
        """TODO: SPECIFY SHAPE OF INPUT VECTOR"""

        """TODO: FIGURE OUT HOW TO IMPLEMENT ENCODER DECODER ARCHITECUTRE"""
        self.copy_bool = copy_bool

        self.test_input = test_input

        self.debug = debug
        self.second_debug = second_debug
        self.save_all_outputs = save_outputs

        self.shape = shape

        """specify dimensions of shape - as in channel length ect. figure out once put it in a dataloader"""

        self.layers = len(test_input)  # number of layers in the encoder.

        self.seq_length = shape[1]

        self.enc_len = len(shape)

        self.input_chans = input_channel_no

        self.hidden_chans = hidden_channel_no

        self.kernel_size = kernel_size

        self.layer_output = layer_output

        # initialise the different conv cells.
        #         self.unit_list = [LSTMunit(input_channel_no, hidden_channel_no, kernel_size) for i in range(self.enc_len)]
        self.dummy_list = [input_channel_no]
        self.dummy_list.extend(
            list(self.test_input)
        )  # allows test input to be an array

        if self.debug:
            print("dummy_list:")
            print(self.dummy_list)

        #         self.unit_list = nn.ModuleList([(LSTMunit(self.dummy_list[i], self.dummy_list[i+1], kernel_size).double()).cuda() for i in range(len(self.test_input))])
        self.unit_list = nn.ModuleList(
            [
                (
                    LSTMunit_t(
                        self.dummy_list[i], self.dummy_list[i + 1], kernel_size
                    ).double()
                )
                for i in range(len(self.test_input))
            ]
        )

        if self.debug:
            print("number of units:")
            print(len(self.unit_list))

    #             print("number of ")

    #         self.unit_list = nn.ModuleList(self.unit_list)

    def forward(self, x, copy_in=False, copy_out=[False, False, False]):
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
        #     def forward(self, x):
        #         copy_in = False
        #         copy_out = [False, False, False]

        #         print("IS X CUDA?")
        #         print(x.is_cuda)
        """loop over layers, then over hidden states

        copy_in is either False or is [[h,c],[h,c]] ect.

        THIS IN NOW CHANGED TO COPY IN

        """

        internal_outputs = []
        """TODO: HOW MANY OUTPUTS TO SAVE"""
        """ S """

        """ TODO: PUT INITIAL ZERO THROUGH THE SYSTEM TO DEFINE H AND C"""

        layer_output = []  # empty list to save each h and c for each step.
        """TODO: DECIDE WHETHER THE ABOVE SHOULD BE ARRAY OR NOT"""

        # x is 5th dimensional tensor.
        # x is of size batch, sequence, layers, height, width

        """TODO: INITIALISE THESE WITH VECTORS."""
        # these need to be of dimensions (batchsizze, hidden_dim, heigh, width)

        size = x.shape

        # need to re arrange the outputs.

        """TODO: SORT OUT H SIZING. """

        batch_size = size[0]
        # change this. h should be of dimensions hidden size, hidden size.
        h_shape = list(
            x.shape[:1] + x.shape[2:]
        )  # seq is second, we miss it with fancy indexing
        h_shape[1] = self.hidden_chans
        if self.debug:
            print("h_shape:")
            print(h_shape)

        # size should be (seq, batch_size, layers, height, weight)

        empty_start_vectors = []

        #### new method of copying vectors. copy_bool, assigned during object
        # construction now deals iwth copying in values.
        # copy in is still used to supply the tensor values.

        k = 0  # to count through our input state list.
        for i in range(self.layers):
            if self.copy_bool[i]:  # if copy bool is true for this layer
                # check purpose of h_shape in below code.
                empty_start_vectors.append(copy_in[k])
                # copies in state for that layer
                """TODO: CHECK IF THIS NEEDS TO BE DETATCHED OR NOT"""
                k += 1  # iterate through input list.

            else:  # i.e if false
                assert self.copy_bool[i] == False, "copy_bool arent bools"

                h_shape = list(
                    x.shape[:1] + x.shape[2:]
                )  # seq is second, we miss it with fancy indexing
                h_shape[1] = self.dummy_list[i + 1]  # check indexing.
                empty_start_vectors.append(
                    [(torch.zeros(h_shape).double()), (torch.zeros(h_shape).double())]
                )

        del k  # clear up k so no spare variables flying about.

        #         for i in range(self.layers):
        #             """CHANGED: NOW HAS COPY IN COPY OUT BASED ON [[0,0][H,C]] FORMAT"""
        #             if copy_in == False: # i.e if no copying in occurs then proceed as normal
        #                 h_shape = list(x.shape[:1] + x.shape[2:]) # seq is second, we miss it with fancy indexing
        #                 h_shape[1] = self.dummy_list[i+1] # check indexing.
        # #                 empty_start_vectors.append([(torch.zeros(h_shape).double()).cuda(), (torch.zeros(h_shape).double()).cuda()])
        #                 empty_start_vectors.append([(torch.zeros(h_shape).double()).cuda(), (torch.zeros(h_shape).double()).cuda()])
        # #             elif copy_in[i] == [0,0]:
        #             elif isinstance(copy_in[i], list):

        #                 assert (len(copy_in) == self.layers), "Length disparity between layers, copy in format"

        #                 # if no copying in in alternate format
        #                 h_shape = list(x.shape[:1] + x.shape[2:]) # seq is second, we miss it with fancy indexing
        #                 h_shape[1] = self.dummy_list[i+1] # check indexing.
        #                 empty_start_vectors.append([(torch.zeros(h_shape).double()).cuda(), (torch.zeros(h_shape).double()).cuda()])

        #             else: # copy in the provided vectors
        #                 assert (len(copy_in) == self.layers), "Length disparity between layers, copy in format"

        #                 """TODO: DECIDE WHETHER TO CHANGE THIS TO AN ASSERT BASED OFF TYPE OF TENSOR."""
        #                 empty_start_vectors.append(copy_in[i])

        #         empty_start_vectors = [[torch.zeros(h_shape), torch.zeros(h_shape)] for i in range(self.layers)]

        if self.debug:
            for i in empty_start_vectors:
                print(i[0].shape)
            print(" \n \n \n")

        #         for i in range(self.layers):
        #             empty_start_vectors.append([torch.tensor()])

        total_outputs = []

        for i in range(self.layers):

            layer_output = []
            if self.debug:
                print("layer iteration:")
                print(i)
            # for each in layer

            """AS WE PUT IN ZEROS EACH TIME THIS MAKES OUR LSTM STATELESS"""
            # initialise with zero or noisy vectors
            # at start of each layer put noisy vector in
            # look at tricks paper to find more effective ideas of how to put this in
            # do we have to initialise with 0 tensors after we go to the second layer
            # or does the h carry over???
            """TODO: REVIEW THIS CHANGE"""

            # copy in for each layer.
            # this is used for encoder decoder architectures.
            # default is to put in empty vectors.

            """TODO: REVIEW THIS SECTION"""
            """CHANGED: TO ALWAYS CHOOSE H AND C"""
            #             if copy_in == False:
            #                 h, c = empty_start_vectors[i]
            #             else: h, c = copy_in[i]

            h, c = empty_start_vectors[i]

            if self.debug:
                print("new h shape")
                print(h.shape)

            """TODO: DO WE HAVE TO PUT BLANK VECTORS IN AT EACH TIMESTEP?"""

            # need to initialise zero states for c and h.
            for j in range(self.seq_length):
                if self.debug:
                    print("inner loop iteration:")
                    print(j)
                if self.debug:
                    print("x dtype is:", x.dtype)
                # for each step in the sequence
                # put x through
                # i.e put through each x value at a given time.

                """TODO: PUT H IN FROM PREVIOUS LAYER, BUT C SHOULD BE ZEROS AT START"""

                if self.debug:
                    print("inner loop size:")
                    print(x[:, j].shape)
                    print("h size:")
                    print(h.shape)

                h, c = self.unit_list[i](x[:, j], h, c)

                # this is record for each output in given layer.
                # this depends whether copying out it enabld
                #                 i
                layer_output.append([h, c])

            """TODO: IMPLEMENT THIS"""
            #             if self.save_all_outputs[i]:
            #                 total_outputs.append(layer_outputs[:,0]) # saves h from each of the layer outputs

            # output
            """OUTSIDE OF SEQ LOOP"""
            """TODO: CHANGE TO NEW OUTPUT METHOD."""
            if copy_out[i] == True:
                # if we want to copy out the contents of this layer:
                internal_outputs.append(layer_output[-1])
                # saves last state and memory which can be subsequently unrolled.
                # when used in an encoder decoder format.
            """removed else statement"""
            #             else:
            #                 internal_outputs.append([0,0])
            # saves null variable so we can check whats being sent out.

            h_output = [
                i[0] for i in layer_output
            ]  # layer_output[:,0] # take h from each timestep.
            if self.debug:
                print("h_output is of size:")
                print(h_output[0].shape)

            """TODO: REVIEW IF 1 IS THE CORRECT AXIS TO CONCATENATE THE VECTORS ALONG"""
            # we now use h as the predictor input to the other layers.
            """TODO: STACK TENSORS ALONG NEW AXIS. """

            x = torch.stack(h_output, 0)
            x = torch.transpose(x, 0, 1)
            if self.second_debug:
                print("x shape in LSTM main:", x.shape)
            if self.debug:
                print("x reshaped dimensions:")
                print(x.shape)

        #         x = torch.zeros(x.shape)
        #         x.requires_grad = True
        #
        #        if len(internal_outputs) == 0:
        #            internal_outputs = [torch.zeros(h_shape, dtype = torch.double, requires_grad = True) for i in range(3)]
        #
        #        internal_outputs = torch.stack(internal_outputs,0) # CHANGED
        return (
            x,
            internal_outputs,
        )  # return new h in tensor form. do we need to cudify this stuff

    def initialise(self):
        """put through zeros to start everything"""


class LSTMencdec_onestep_t(nn.Module):
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

    def __init__(self, structure, input_channels, kernel_size=5, debug=False):
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

        super(LSTMencdec_onestep_t, self).__init__()
        assert (
            len(structure.shape) == 2
        ), "structure should be a 2d numpy array with two rows"

        self.debug = debug

        # shape of input
        shape = [1, 10, 3, 16, 16]

        self.structure = structure
        self.input_channels = input_channels
        self.kernel_size = kernel_size

        # extract arguments for encoder and decoder constructors from structure
        (
            self.enc_shape,
            self.dec_shape,
            self.enc_copy_out,
            self.dec_copy_in,
        ) = self.input_test()

        if self.debug:
            print("enc_shape, dec_shape, enc_copy_out, dec_copy_in:")
            print(self.enc_shape)
            print(self.dec_shape)
            print(self.enc_copy_out)
            print(self.dec_copy_in)

        # why does this have +1 at third input and decoder hasnt??????

        # initialise encoder and decoder structures
        self.encoder = LSTMmain_t(
            shape,
            self.input_channels,
            len(self.enc_shape) + 1,
            self.kernel_size,
            layer_output=self.enc_copy_out,
            test_input=self.enc_shape,
            copy_bool=[False for k in range(len(self.enc_shape))],
        )

        # now one step in sequence
        shape = [1, 1, 1, 64, 64]

        self.decoder = LSTMmain_t(
            shape,
            self.enc_shape[-1],
            len(self.dec_shape),
            self.kernel_size,
            layer_output=1,
            test_input=self.dec_shape,
            copy_bool=self.dec_copy_in,
            second_debug=False,
        )

        # initialise encoder and decoder network

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
        enc_shape = enc_layer[enc_layer != 0]
        dec_layer = self.structure[1]
        dec_shape = dec_layer[dec_layer != 0]

        # find vertical overlaps between non zero elements
        for i in range(len(enc_layer)):
            if self.debug:
                print(enc_layer[i], dec_layer[i])
            if (enc_layer[i] != 0) and (dec_layer[i] != 0):
                copy_grid.append(True)
            else:
                copy_grid.append(False)

        enc_overlap = copy_grid[: len(enc_layer) - 1]

        num_dec_zeros = len(dec_layer[dec_layer == 0])

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
        x, out_states = self.encoder(x, copy_in=False, copy_out=self.enc_copy_out)

        # select last input of encoder for conditional encoder decoder model.
        x = x[:, -1:, :, :, :]

        res, _ = self.decoder(
            x, copy_in=out_states, copy_out=[False, False, False, False, False]
        )
        if self.debug:
            print("FINISHING ONE PASS")
        return res
