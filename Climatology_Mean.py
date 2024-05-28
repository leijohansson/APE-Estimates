# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:28:59 2024

@author: Linne

calculating mean climatology (averaged over months)
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
#grid spacing 0.25 deg
import scipy.io
import numpy.ma as ma
import pickle 

file1 = f'WOCE_Data/APEarrays/WAGHC_APE_PYC-01.npy'
shape = np.load(file1).shape

for method in ['BAR', 'PYC']:
    timespace_arr = np.zeros(shape)
    timeidx = -1
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
        APE_all = np.load(file)
        timespace_arr += APE_all
    
    #finding time mean (over year) climatology
    mean = timespace_arr/12
    np.save(f'WOCE_Data/APEarrays/WAGHC_APE_{method}-mean.npy', mean)


