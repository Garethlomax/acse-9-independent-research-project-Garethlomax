# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:41:26 2019

@author: Gareth
"""

import seaborn as sns

sns.set()

# Load the example iris dataset
# planets = sns.load_dataset("ucdp_")

# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
# ax = sns.scatterplot(x="deaths_a", y="deaths_b",
#                     hue="type_of_violence", size="deaths_civilians",
#                     sizes=(10, 200),
#                     data=africa)


g = sns.FacetGrid(
    africa,
    hue="latitude",
    subplot_kws=dict(projection="polar"),
    height=4.5,
    sharex=False,
    sharey=False,
    despine=False,
)

# Draw a scatterplot onto each axes in the grid
g.map(sns.scatterplot, "longitude", "best")
