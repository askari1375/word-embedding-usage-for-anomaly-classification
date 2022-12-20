# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:33:30 2022

@author: Amirhossein
"""

import numpy as np
import matplotlib.pyplot as plt



y_pred = np.load("results/anomaly predictions.npy")
y_true = np.load("results/anomaly true values.npy")

d2 = np.square(y_pred - y_true)

total_rsme = np.sqrt(np.mean(d2))

axial_rsme = np.sqrt(np.mean(d2, axis = 0))

binwidth = 0.01
hist_bins = np.arange(min(axial_rsme), max(axial_rsme) + binwidth, binwidth)
plt.hist(axial_rsme, bins = hist_bins)

print(total_rsme)