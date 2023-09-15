# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 22:39:38 2019

@author: Gareth
"""


def batch_loss_histogram(model, train_loader, loss_func):

    model.eval()
    # calculate x and prediction
    for a, b in train_loader:
        # a in input, b is truth
        break  # train loader cannot be indexed

    with torch.no_grad():
        x = model(a.cuda())

        x = x.cpu()
        #     print(x.shape)
        # now over each one in x - we do
        # loss_func = nn.BCEWithLogitsLoss()
        loss = []
        for i in range(len(x)):
            loss.append(loss_func(x[i, :, 0], b[i : i + 1]).item())

    return loss
