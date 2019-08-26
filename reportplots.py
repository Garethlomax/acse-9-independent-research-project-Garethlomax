# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 00:51:45 2019

@author: Gareth
"""

import seaborn as sns
import pandas as pd

data = pd.read_csv("bce_graph_data.csv")

sns.lineplot(data = data[['BCE_1', 'BCE_2', 'BCE_3']])
