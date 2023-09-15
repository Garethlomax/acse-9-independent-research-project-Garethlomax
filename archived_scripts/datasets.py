# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:45:06 2019

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


class HDF5Dataset(Dataset):
    """dataset wrapper for hdf5 dataset to allow for lazy loading of data. This
    allows ram to be conserved.

    As the hdf5 dataset is not partitioned into test and validation, the dataset
    takes a shuffled list of indices to allow specification of training and
    validation sets.

    MAKE SURE TO CALL DEL ON GENERATED OBJECTS OTHERWISE WE WILL CLOG UP RAM

    """

    def __init__(self, path, index_map, transform=None):

        #        %cd /content/drive/My \Drive/masters_project/data
        # changes directory to the one where needed.

        self.path = path

        self.index_map = index_map  # maps to the index in the validation split
        # due to hdf5 lazy loading index map must be in ascending order.
        # this may be an issue as we should shuffle our dataset.
        # this will be raised as an issue as we consider a work around.
        # we should keep index map shuffled, and take the selection from the
        # shuffled map and select in ascending order.

        self.file = h5py.File(path, "r")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):

        i = self.index_map[i]  # index maps from validation set to select new orders
        #         print(i)
        if isinstance(i, list):  # if i is a list.
            i.sort()  # sorts into ascending order as specified above

        """TODO: CHECK IF THIS RETURNS DOUBLE"""

        predictor = torch.tensor(self.file["predictor"][i])

        truth = torch.tensor(self.file["truth"][i])

        return predictor, truth


def initialise_dataset_HDF5(valid_frac=0.1, dataset_length=9000):
    """
    Returns datasets for training and validation.

    Loads in datasets segmenting for validation fractions.



    """

    if valid_frac != 0:

        dummy = np.array(range(dataset_length))  # clean this up - not really needed

        train_index, valid_index = validation_split(
            dummy, n_splits=1, valid_fraction=0.1, random_state=0
        )

        train_dataset = HDF5Dataset("train_set.hdf5", index_map=train_index)

        valid_dataset = HDF5Dataset("test_set.hdf5", index_map=valid_index)

        return train_dataset, valid_dataset

    else:
        print("not a valid fraction for validation")  # turn this into an assert.


def initialise_dataset_HDF5_full(
    dataset,
    valid_frac=0.1,
    dataset_length=9000,
    avg=0,
    std=0,
    application_boolean=[0, 0, 0, 0, 0],
):
    """
    Returns datasets for training and validation.

    Loads in datasets segmenting for validation fractions.



    """

    if valid_frac != 0:

        dummy = np.array(range(dataset_length))  # clean this up - not really needed

        train_index, valid_index = validation_split(
            dummy, n_splits=1, valid_fraction=0.1, random_state=0
        )

        train_index = list(train_index)

        valid_index = list(valid_index)

        train_dataset = HDF5Dataset_with_avgs(
            dataset, train_index, avg, std, application_boolean
        )

        valid_dataset = HDF5Dataset_with_avgs(
            dataset, valid_index, avg, std, application_boolean
        )

        return train_dataset, valid_dataset

    else:
        print("not a valid fraction for validation")  # turn this into an assert.


def validation_split(data, n_splits=1, valid_fraction=0.1, random_state=0):
    """
    Function to produce a validation set from test set.
    THIS SHUFFLES THE SAMPLES. __NOT__ THE SEQUENCES.
    """
    dummy_array = np.zeros(len(data))
    split = StratifiedShuffleSplit(n_splits, test_size=valid_fraction, random_state=0)
    generator = split.split(torch.tensor(dummy_array), torch.tensor(dummy_array))
    return [(a, b) for a, b in generator][0]


def unsqueeze_data(data):
    """
    Takes in moving MNIST object - must then account for
    """

    # split moving mnist data into predictor and ground truth.
    predictor = data[:][0].unsqueeze(2)
    predictor = predictor.double()

    truth = data[:][1].unsqueeze(2)  # this should be the moving mnist sent in
    truth = truth.double()

    return predictor, truth
    # the data should now be unsqueezed.


#
# def initialise_dataset(data):
#    # unsqueeze data, adding a channel dimension for later convolution.
#    # this also gets rid of the annoying tuple format
#    predictor, truth = unsqueeze_data(data)
#
#    train_index, valid_index = validation_split(data)
#
#    train_predictor = predictor[train_index]
#    valid_predictor = predictor[valid_index]
#
#    train_truth = truth[train_index]
#    valid_truth = truth[valid_index]
#
#    train_dataset = SequenceDataset(train_predictor, train_truth)
#    valid_dataset = SequenceDataset(valid_predictor, valid_truth)
#
#    return train_dataset, valid_dataset

# def comb_loss_func(pred, y):
#    """hopefully should work like kl and bce for VAE"""
#    mse = nn.MSELoss()
#    ssim = pytorch_ssim.SSIM()
#    mse_loss = mse(pred, y[:,:1,:,:,:])
#    ssim_loss = -ssim(pred[:,0,:,:,:], y[:,0,:,:,:])
#    return mse_loss + ssim_loss


def train_enc_dec(model, optimizer, dataloader, loss_func=nn.MSELoss()):
    """
    training function

    by default mseloss

    could try brier score.

    """
    i = 0
    model.train()  # enables training for model.
    tot_loss = 0
    for x, y in dataloader:
        #         print("training")
        x = x.to(device)  # send to cuda.
        y = y.to(device)
        optimizer.zero_grad()  # zeros saved gradients in the optimizer.
        # prevents multiple stacking of gradients
        # this is important to do before we evaluate the model as the
        # model is currenly in model.train() mode

        prediction = model(x)  # x should be properly formatted - of size
        """THIS DOESNT DEAL WITH SEQUENCE LENGTH VARIANCE OF PREDICTION OR Y"""

        #         print("the size of prediction is:", prediction.shape)
        # last image sequence.

        """ACTUAL FUNCTION THATS BEEN COMMENTED OUT."""
        #         loss = loss_func(prediction, y[:,:1,:,:,:])
        """CHANGED BECAUSE """
        print(prediction.shape)
        print(y.shape)
        loss = loss_func(prediction[:, 0, 0], y)

        #         loss = comb_loss_func(prediction, y)
        #         print(prediction.shape)
        #         print(y[:,:1,:,:,:].shape)
        """commented out """
        #         loss = - loss_func(prediction[:,0,:,:,:], y[:,0,:,:,:])

        # ssim_out = -ssim_loss(train[0][0][-1:],  x[0])
        # ssim_value = - ssim_out.data

        loss.backward()  # differentiates to find minimum.
        #         printm()

        ##

        # implement the interpreteable stuff here.
        # as it is very unlikely we predict every pixel correctly we will not
        # use accuracy.
        # technically this is a regression problem, not a classification.

        optimizer.step()  # steps forward the optimizer.
        # uses loss.backward() to give gradient.
        # loss is negative.
        #         del x # make sure the garbage is collected.
        #         del y
        """commented it out"""
        tot_loss += loss.item()  # .data.item()
        print("BATCH:")
        print(i)
        i += 1
        #         if i == 20:
        #             break
        print("MSE_LOSS:", tot_loss / i)
    return model, tot_loss / i  # trainloss, trainaccuracy


def validate(model, dataloader, loss_func=nn.MSELoss()):

    """as for train_enc_dec but without training - and acting upon validation
    data set
    """
    tot_loss = 0
    i = 0
    model.eval()  # puts out of train mode so we do not mess up our gradients
    for x, y in dataloader:
        with torch.no_grad():  # no longer have to specify tensors
            # as volatile = True. as of modern pytorch use torch.no_grad.

            x = x.to(device)  # send to cuda. need to change = sign as to(device)
            y = y.to(device)  # produces a copy on thd gpu not moves it.
            prediction = model(x)

            loss = loss_func(prediction[:, 0, 0], y)

            tot_loss += loss.item()
            i += 1

            print("MSE_VALIDATION_LOSS:", tot_loss / i)

    return tot_loss / i  # returns total loss averaged across the dataset.


def train_main(model, params, train, valid, epochs=30, batch_size=1):
    # make sure model is ported to cuda
    # make sure seed has been specified if testing comparative approaches

    #     if model.is_cuda == False:
    #         model.to(device)

    # initialise optimizer on model parameters
    # chann
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, amsgrad=True)
    loss_func = nn.MSELoss()
    #     loss_func = nn.BCELoss()
    #     loss_func = pytorch_ssim.SSIM()

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True
    )  # implement moving MNIST data input
    validation_loader = DataLoader(
        valid, batch_size=batch_size, shuffle=False
    )  # implement moving MNIST

    for epoch in range(epochs):

        train_enc_dec(model, optimizer, train_loader, loss_func=loss_func)  # changed

        torch.save(
            optimizer.state_dict(), f"Adam_new_ams_changed" + str(epoch) + ".pth"
        )
        torch.save(model.state_dict(), f"Test_new_ams_changed" + str(epoch) + ".pth")

    #         validate(model, validation_loader)

    return model, optimizer


class HDF5Dataset_with_avgs(Dataset):
    """dataset wrapper for hdf5 dataset to allow for lazy loading of data. This
    allows ram to be conserved.

    As the hdf5 dataset is not partitioned into test and validation, the dataset
    takes a shuffled list of indices to allow specification of training and
    validation sets.

    MAKE SURE TO CALL DEL ON GENERATED OBJECTS OTHERWISE WE WILL CLOG UP RAM

    """

    def __init__(self, path, index_map, avg, std, application_boolean, transform=None):

        #        %cd /content/drive/My \Drive/masters_project/data
        # changes directory to the one where needed.

        self.path = path

        self.index_map = index_map  # maps to the index in the validation split
        # due to hdf5 lazy loading index map must be in ascending order.
        # this may be an issue as we should shuffle our dataset.
        # this will be raised as an issue as we consider a work around.
        # we should keep index map shuffled, and take the selection from the
        # shuffled map and select in ascending order.
        self.avg = avg
        self.std = std
        self.application_boolean = application_boolean

        self.file = h5py.File(path, "r")

    #         for i in range(len(application_boolean)):
    #             # i.e gaussian transformation doesnt happen. (x - mu / sigma)
    #             if application_boolean == 0:
    #                 self.avg[i] = 0
    #                 self.std[i] = 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):

        i = self.index_map[i]  # index maps from validation set to select new orders
        #         print(i)
        if isinstance(i, list):  # if i is a list.
            i.sort()  # sorts into ascending order as specified above

        """TODO: CHECK IF THIS RETURNS DOUBLE"""

        predictor = torch.tensor(self.file["predictor"][i])
        #         print("predictor shape:", predictor.shape)
        # is of batch size, seq length,

        truth = torch.tensor(self.file["truth"][i])
        #         print("truth shape:", truth.shape)
        # only on layer so not in loop.
        #         truth -= self.avg[0]
        #         truth /= self.std[0]

        if isinstance(i, list):
            for j in range(len(self.avg)):
                if self.application_boolean[j]:
                    predictor[:, :, j] -= self.avg[j]
                    predictor[:, :, j] /= self.std[j]

        else:
            for j in range(len(self.avg)):
                if self.application_boolean[j]:
                    predictor[:, j] -= self.avg[j]
                    predictor[:, j] /= self.std[j]

        #             #i.e if we are returning a single index.
        # #         # the value of truth should be [0] in the predictor array.
        #         for j in range(len(self.avg)):
        #             if self.application_boolean[j]:
        #                 predictor[:,:,j] -= self.avg[j]
        #                 predictor[:,:,j] /= self.std[j]

        #                 # sort out dimensions of truth at some point

        return predictor, truth
