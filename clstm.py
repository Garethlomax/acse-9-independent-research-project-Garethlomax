# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:45:04 2019

@author: Gareth
"""

## imports
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


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


## LSTMunit -- class.
class LSTMunit(nn.Module):
    def __init__(self, input_channel_no, hidden_channels_no, kernel_size, stride = 1):
        super(LSTMunit, self).__init__()
        """base unit for an overall convLSTM structure. convLSTM exists in keras but
        not pytorch. LSTMunit repersents one cell in an overall convLSTM encoder decoder format
        the structure of convLSTMs lend themselves well to compartmentalising the LSTM
        cells.

        Each cell takes an input the data at the current timestep Xt, and a hidden
        representation from the previous timestep Ht-1

        Each cell outputs Ht
        """


        self.input_channels = input_channel_no

        self.output_channels = hidden_channels_no

        self.kernel_size = kernel_size

        self.padding = (int((self.kernel_size - 1) / 2 ), int((self.kernel_size - 1) / 2 ))#to ensure output image same dims as input
        # as in conv nowcasting - see references
        self.stride = stride # for same reasons as above


        # cannot do this because of the modules not being registered when stored in a list
        """TODO: CHANGE THIS LAYOUT OF CONVOLUTIONAL LAYERS. """
        # list of filter names
        self.filter_name_list = ['Wxi', 'Wxf', 'Wxc', 'Wxo','Whi', 'Whf', 'Whc', 'Who']

        """ TODO : DEAL WITH BIAS HERE. """
        """ TODO: CAN INCLUDE BIAS IN ONE OF THE CONVOLUTIONS BUT NOT ALL OF THEM - OR COULD INCLUDE IN ALL? """

        # first add convolutions without bias
        self.conv_list = [nn.Conv2d(self.input_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = False).cuda() for i in range(4)]

        # now with bias.
        self.conv_list = self.conv_list + [(nn.Conv2d(self.output_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = True).cuda()).double() for i in range(4)]

        self.conv_dict = nn.ModuleDict(zip(self.filter_name_list, self.conv_list))

        """ TODO: decide whether this should be put into function. """
        """TODO: put correct dimensions of tensor in shape"""
        """TODO: DEFINE THESE SYMBOLS. """
        """TODO: PUT THIS IN CONSTRUCTOR."""

        # of dimensions seq length, hidden layers, height, width
        # has n_output channels as is multiplied after the input has been convolved.
        shape = [1, self.output_channels, 16, 16]

        # not convolutions as multiplicative - see origional paper
        self.Wco = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)
        self.Wcf = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)
        self.Wci = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)

        # activation functions.
        self.tanh = torch.tanh
        self.sig  = torch.sigmoid

    def forward(self, x, h, c):
        """ put the various nets in here - instanciate the other convolutions."""
        """TODO: SORT BIAS OUT HERE"""
        """TODO: PUT THIS IN SELECTOR FUNCTION? SO ONLY PUT IN WXI ECT TO MAKE EASIER TO DEBUG?"""

        # following the definition from origonal paper
        i_t = self.sig(self.conv_dict['Wxi'](x) + self.conv_dict['Whi'](h) + self.Wci * c)
        f_t = self.sig(self.conv_dict['Wxf'](x) + self.conv_dict['Whf'](h) + self.Wcf * c)
        c_t = f_t * c + i_t * self.tanh(self.conv_dict['Wxc'](x) + self.conv_dict['Whc'](h))
        o_t = self.sig(self.conv_dict['Wxo'](x) + self.conv_dict['Who'](h) + self.Wco * c_t)
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

    def copy_in(self):
        """dummy function to copy in the internals of the output in the various architectures i.e encoder decoder format"""


"""# lstm full unit"""

"""TODO: IMPORTANT
WHEN COPYING STATES OVER, INITIAL STATE OF DECODER IS BOTH LAST H AND LAST C
FROM THE LSTM BEING COPIED FROM.

WE ALSO NEED TO INCLUDE THE ABILITY TO OUTPUT THE LAST H AND C AT EACH TIMESTEP
AS INPUT.
"""


""" SEQUENCE, BATCH SIZE, LAYERS, HEIGHT, WIDTH"""

##############################################################################
class LSTMunit_t(nn.Module):
    def __init__(self, input_channel_no, hidden_channels_no, kernel_size, stride = 1):
        super(LSTMunit_t, self).__init__()
        """base unit for an overall convLSTM structure. convLSTM exists in keras but
        not pytorch. LSTMunit repersents one cell in an overall convLSTM encoder decoder format
        the structure of convLSTMs lend themselves well to compartmentalising the LSTM
        cells.

        Each cell takes an input the data at the current timestep Xt, and a hidden
        representation from the previous timestep Ht-1

        Each cell outputs Ht
        """


        self.input_channels = input_channel_no

        self.output_channels = hidden_channels_no

        self.kernel_size = kernel_size

        self.padding = (int((self.kernel_size - 1) / 2 ), int((self.kernel_size - 1) / 2 ))#to ensure output image same dims as input
        # as in conv nowcasting - see references
        self.stride = stride # for same reasons as above

        # need convolutions, cells, tanh, sigmoid?
        # need input size for the lstm - on size of layers.
        # cannot do this because of the modules not being registered when stored in a list
        # can if we convert it to a parameter dict

        # list of names of filter to put in dictionary.
        # some of these are not convolutions
        """TODO: CHANGE THIS LAYOUT OF CONVOLUTIONAL LAYERS. """



        self.filter_name_list = ['Wxi', 'Wxf', 'Wxc', 'Wxo','Whi', 'Whf', 'Whc', 'Who']

        """ TODO : DEAL WITH BIAS HERE. """
        """ TODO: CAN INCLUDE BIAS IN ONE OF THE CONVOLUTIONS BUT NOT ALL OF THEM - OR COULD INCLUDE IN ALL? """

        # list of concolution instances for each lstm cell step
       #  nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=0),
        self.conv_list = [nn.Conv2d(self.input_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = False) for i in range(4)]
#         self.conv_list = [nn.Conv2d(self.input_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = False) for i in range(4)]

#         self.conv_list = self.conv_list + [(nn.Conv2d(self.output_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = True)).double() for i in range(4)]

        self.conv_list = self.conv_list + [(nn.Conv2d(self.output_channels, self.output_channels, kernel_size =  self.kernel_size, stride = self.stride, padding = self.padding, bias = True)).double() for i in range(4)]
#         self.conv_list = nn.ModuleList(self.conv_list)
        # stores nicely in dictionary for compact readability.
        # most ML code is uncommented and utterly unreadable. Here we try to avoid this
        self.conv_dict = nn.ModuleDict(zip(self.filter_name_list, self.conv_list))

        # may be able to combine all the filters and combine all the things to be convolved - as long as there is no cross layer convolution
        # technically the filter will be the same? - check this later.

        # set up W_co, W_cf, W_co as variables.
        """ TODO: decide whether this should be put into function. """


        """TODO: put correct dimensions of tensor in shape"""

        # of dimensions seq length, hidden layers, height, width
        """TODO: DEFINE THESE SYMBOLS. """
        """TODO: PUT THIS IN CONSTRUCTOR."""
        shape = [1, self.output_channels, 16, 16]
        # not convolutions as multiplicative pridu
#        self.Wco = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)
#        self.Wcf = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)
#        self.Wci = nn.Parameter((torch.zeros(shape).double()).cuda(), requires_grad = True)


        self.Wco = nn.Parameter((torch.zeros(shape).double()), requires_grad = True)
        self.Wcf = nn.Parameter((torch.zeros(shape).double()), requires_grad = True)
        self.Wci = nn.Parameter((torch.zeros(shape).double()), requires_grad = True)
#         self.Wco.name = "test"
#         self.Wco = torch.zeros(shape, requires_grad = True).double()
#         self.Wcf = torch.zeros(shape, requires_grad = True).double()
#         self.Wci = torch.zeros(shape, requires_grad = True).double()

        # activation functions.
        self.tanh = torch.tanh
        self.sig  = torch.sigmoid

#     (1, 6, kernel_size=5, padding=2, stride=1).double()
    def forward(self, x, h, c):
        """ put the various nets in here - instanciate the other convolutions."""
        """TODO: SORT BIAS OUT HERE"""
        """TODO: PUT THIS IN SELECTOR FUNCTION? SO ONLY PUT IN WXI ECT TO MAKE EASIER TO DEBUG?"""
#         print("size of x is:")
#         print(x.shape)
        # ERROR IS IN LINE 20
        #print(self.conv_dict['Wxi'](x).shape)
#         print("X:")
#         print(x.is_cuda)
#         print("H:")
#         print(h.is_cuda)
#         print("C")
#         print(c.is_cuda)

        i_t = self.sig(self.conv_dict['Wxi'](x) + self.conv_dict['Whi'](h) + self.Wci * c)
        f_t = self.sig(self.conv_dict['Wxf'](x) + self.conv_dict['Whf'](h) + self.Wcf * c)
        c_t = f_t * c + i_t * self.tanh(self.conv_dict['Wxc'](x) + self.conv_dict['Whc'](h))
        o_t = self.sig(self.conv_dict['Wxo'](x) + self.conv_dict['Who'](h) + self.Wco * c_t)
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

    def copy_in(self):
        """dummy function to copy in the internals of the output in the various architectures i.e encoder decoder format"""

"""# lstm full unit"""

"""TODO: IMPORTANT
WHEN COPYING STATES OVER, INITIAL STATE OF DECODER IS BOTH LAST H AND LAST C
FROM THE LSTM BEING COPIED FROM.

WE ALSO NEED TO INCLUDE THE ABILITY TO OUTPUT THE LAST H AND C AT EACH TIMESTEP
AS INPUT.
"""


""" SEQUENCE, BATCH SIZE, LAYERS, HEIGHT, WIDTH"""


###############################################################################
class LSTMmain(nn.Module):


    """ collection of units to form encoder/ decoder branches - decide which are which
    need funcitonality to copy in and copy out outputs.


    layer output is array of booleans selectively outputing for each layer i.e
    for three layer can have output on second and third but not first with
    layer_output = [0,1,1]"""

    """TODO: DECIDE ON OUTPUT OF HIDDEN CHANNEL LIST """
    def __init__(self, shape, input_channel_no, hidden_channel_no, kernel_size, layer_output, test_input, copy_bool = False, debug = False, save_outputs = True, decoder = False, second_debug = False):
        super(LSTMmain, self).__init__()

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

        self.layers = len(test_input) #number of layers in the encoder.

        self.seq_length = shape[1]

        self.enc_len = len(shape)

        self.input_chans = input_channel_no

        self.hidden_chans = hidden_channel_no

        self.kernel_size = kernel_size

        self.layer_output = layer_output

        # initialise the different conv cells.
        self.dummy_list = [input_channel_no] + list(self.test_input) # allows test input to be an array

        if self.debug:
            print("dummy_list:")
            print(self.dummy_list)

        self.unit_list = nn.ModuleList([(LSTMunit(self.dummy_list[i], self.dummy_list[i+1], kernel_size).double()).cuda() for i in range(len(self.test_input))])

        if self.debug:
            print("number of units:")
            print(len(self.unit_list))
#             print("number of ")

    def forward(self, x, copy_in = False, copy_out = [False, False, False]):

        """loop over layers, then over hidden states

        copy_in is either False or is [[h,c],[h,c]] ect.

        THIS IN NOW CHANGED TO COPY IN

        """

        """TODO: HOW MANY OUTPUTS TO SAVE"""
        """ S """

        """ TODO: PUT INITIAL ZERO THROUGH THE SYSTEM TO DEFINE H AND C"""

        """TODO: DECIDE WHETHER THE ABOVE SHOULD BE ARRAY OR NOT"""

        """TODO: INITIALISE THESE WITH VECTORS."""

        """TODO: SORT OUT H SIZING. """


        internal_outputs = []


        layer_output = [] # empty list to save each h and c for each step.


        # x is 5th dimensional tensor.
        # x is of size batch, sequence, layers, height, width


        # these need to be of dimensions (batchsizze, hidden_dim, heigh, width)


        # change this. h should be of dimensions hidden size, hidden size.

        # finds dimension of h
        h_shape = list(x.shape[:1] + x.shape[2:]) # seq is second, we miss it with fancy indexing
        h_shape[1] = self.hidden_chans
        if self.debug:
            print("h_shape:")
            print(h_shape)

        # size should be (seq, batch_size, layers, height, weight)
        empty_start_vectors = []

        # copies in or instances the hidden and cell states h, c
        k = 0 # to count through our input state list.
        for i in range(self.layers): # for each lstm layer in the deep lstm.
            if self.copy_bool[i]:


                # this layer copies in from an encoder hidden state
                empty_start_vectors.append(copy_in[k])
                # copies in state for that layer
                # layer is not detatched to allow convergence of encoder unit

                k += 1 # iterate through input list.

            else: # no copy in for this layer
                assert self.copy_bool[i] == False, "copy_bool arent bools"

                h_shape = list(x.shape[:1] + x.shape[2:])
                h_shape[1] = self.dummy_list[i+1] # check indexing.
                empty_start_vectors.append([(torch.zeros(h_shape).double()).cuda(), (torch.zeros(h_shape).double()).cuda()])

                # each initial new internal state should be re instanced to avoid
                # cross minibatch computational tree connections
                # initialised as 0 as gaussian noise has been shown to contribute
                # to exploding gradients

        del k # clear up k so no spare variables flying about.

        if self.debug:
            for i in empty_start_vectors:
                print(i[0].shape)
            print(" \n \n \n")

#

#        total_outputs = []

        # main iteration loop
        for i in range(self.layers):
            layer_output = []
            if self.debug:
                print("layer iteration:")
                print(i)

            """AS WE PUT IN ZEROS EACH TIME THIS MAKES OUR LSTM STATELESS"""
            """TODO: REVIEW THIS CHANGE"""
            """TODO: REVIEW THIS SECTION"""
            """CHANGED: TO ALWAYS CHOOSE H AND C"""

            h, c = empty_start_vectors[i]

            if self.debug:
                print("new h shape")
                print(h.shape)

            """TODO: DO WE HAVE TO PUT BLANK VECTORS IN AT EACH TIMESTEP?"""
            for j in range(self.seq_length): # for each sequence
                if self.debug:
                    print("inner loop iteration:")
                    print(j)
                if self.debug:
                    print("x dtype is:" , x.dtype)
                # for each step in the sequence
                # put x through
                # i.e put through each x value at a given time.

                """TODO: PUT H IN FROM PREVIOUS LAYER, BUT C SHOULD BE ZEROS AT START"""

                if self.debug:
                    print("inner loop size:")
                    print(x[:,j].shape)
                    print("h size:")
                    print(h.shape)

                # passes in the sequence
                h, c = self.unit_list[i](x[:,j], h, c)


                #saves h, c to pass to next layer
                layer_output.append([h, c])

            """TODO: IMPLEMENT THIS"""
            """OUTSIDE OF SEQ LOOP"""
            """TODO: CHANGE TO NEW OUTPUT METHOD."""
            #copies out for each layer
            if copy_out[i] == True:
                internal_outputs.append(layer_output[-1])
                # saves last state and memory which can be subsequently unrolled
                # by decoder when used in an encoder decoder format.

            # passes on internal representation to next layer
            h_output = [i[0] for i in layer_output]

            if self.debug:
                print("h_output is of size:")
                print(h_output[0].shape)


            #combine lists into deep tensors
            x = torch.stack(h_output,0)
            x = torch.transpose(x, 0, 1)

            if self.second_debug:
                print("x shape in LSTM main:" , x.shape)
            if self.debug:
                print("x reshaped dimensions:")
                print(x.shape)

        return x , internal_outputs # return new h in tensor form.

    def initialise(self):
        """put through zeros to start everything"""


##################################################################################

##################################################################################

class LSTMencdec_onestep(nn.Module):
    """structure is overall architecture of """
    def __init__(self, structure, input_channels, kernel_size = 5, debug = True):
        super(LSTMencdec_onestep, self).__init__()
#         assert isinstance(structure, np.array), "structure should be a 2d numpy array"
        assert len(structure.shape) == 2, "structure should be a 2d numpy array with two rows"
        self.debug = debug

        """TODO: MAKE KERNEL SIZE A LIST SO CAN SPECIFY AT EACH JUNCTURE."""
        shape = [1,10,3,16,16]

        self.structure = structure
        """STRUCTURE IS AN ARRAY - CANNOT USE [] + [] LIST CONCATENATION - WAS ADDING ONE ONTO THE ARRAY THING."""
        self.input_channels = input_channels
        self.kernel_size = kernel_size

        """TODO: ASSERT THAT DATATYPE IS INT."""

        # extract encoder and decoder dimensions from structure input format
        self.enc_shape, self.dec_shape, self.enc_copy_out, self.dec_copy_in = self.input_test()

        if self.debug:
            print("enc_shape, dec_shape, enc_copy_out, dec_copy_in:")
            print(self.enc_shape)
            print(self.dec_shape)
            print(self.enc_copy_out)
            print(self.dec_copy_in)




         # why does this have +1 at third input and decoder hasnt??????
        # initialise encoder
        self.encoder = LSTMmain(shape, self.input_channels, len(self.enc_shape)+1, self.kernel_size, layer_output = self.enc_copy_out, test_input = self.enc_shape, copy_bool = [False for k in range(len(self.enc_shape))]).cuda()

        # now one step in sequence
        shape = [1,1,1,64,64]
        # initialise decoder
        self.decoder = LSTMmain(shape, self.enc_shape[-1], len(self.dec_shape), self.kernel_size, layer_output = 1, test_input = self.dec_shape, copy_bool = self.dec_copy_in,  second_debug = False).cuda()

    def input_test(self):
        """check input structure to make sure there is overlap between encoder
        and decoder.
        """
        copy_grid = []
        # finds dimensions of the encoder
        enc_layer = self.structure[0]
        enc_shape = enc_layer[enc_layer!=0]
        dec_layer = self.structure[1]
        dec_shape = dec_layer[dec_layer!=0]
#





        #set up boolean grid of where the overlaps are.
        for i in range(len(enc_layer)):
            if self.debug:
                print(enc_layer[i], dec_layer[i])
            if (enc_layer[i] != 0) and (dec_layer[i] != 0):
                copy_grid.append(True)
            else:
                copy_grid.append(False)


        enc_overlap = copy_grid[:len(enc_layer)-1]

        num_dec_zeros = len(dec_layer[dec_layer==0]) # will this break if no zeros?

        dec_overlap = copy_grid[num_dec_zeros:]

        return enc_shape, dec_shape, enc_overlap, dec_overlap



    def forward(self, x):

        x, out_states = self.encoder(x, copy_in = False, copy_out = self.enc_copy_out)

        # first input to the encoder is previous output from the encoder
        # meaning the encoder is conditional
        x = x[:,-1:,:,:,:]

        res, _ = self.decoder(x, copy_in = out_states, copy_out = [False, False, False,False, False])
        print("FINISHING ONE PASS")
        return res

###################################################################################





