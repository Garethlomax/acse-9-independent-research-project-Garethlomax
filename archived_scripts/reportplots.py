# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:51:45 2019

@author: Gareth
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
sns.set_style("ticks")
# sns.set_style("spine")
data = pd.read_csv("bce_graph_data.csv")

plot = sns.lineplot(data=data[["BCE_1", "BCE_2", "BCE_3", "BCE_4"]])
plot.set_xlim(0, 105)
plot.set_xlabel("Epochs")
plot.set_ylabel("Validation set loss")
plot.set_title(
    "Comparing the effects of Multilabel binary cross entropy weight scaling"
)
# plot.set_xlim(0,105)
