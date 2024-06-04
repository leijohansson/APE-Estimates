# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:09:32 2024

@author: Linne
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
#grid spacing 0.25 deg
import scipy.io
import numpy.ma as ma
import pickle 
import pandas as pd

datadir = datapath + 'Data\\' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(datadir+filename)
shape = data.temperature.squeeze().shape

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.salinity.squeeze().to_numpy()
vol_nans = volume_s/volume_s

#finding vertical distance represented by each grid point
depths = data.depth.to_numpy()

# for year in range(startyear, endyear+1):
for year in [2019]:
    # for month in range(1, 13):
    for month in [1]:
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = f'APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(datapath + file)
        APE_all = APE_all*vol_nans
        zonalmean = np.nanmean(APE_all, axis = 2)
        log10zm = np.log10(zonalmean)
        flipped = np.flip(log10zm, axis =0)
        #plotting vertically integrated APE per unit area
        fig, ax = plt.subplots(figsize = (20, 15))
        ax.set_facecolor('darkgrey')
        X, Y = np.meshgrid(data.lat, data.depth)
        plot = ax.contourf(X, Y, log10zm, levels = 15)
        # plt.gca().flip
        plt.colorbar(plot, label = '$log_{10}$ zonal mean APE density $(Jm^{-3})$', location = 'bottom')
        ax.set_title(f'EN4 {year}-{month} Zonal Mean')
        ax.set_ylabel('Depth, m')
        ax.set_xlabel('Latitude, $^\circ$')
        ax.invert_yaxis()
        plt.savefig(f'EN4 Plots\zonalmean_log10APE.png', bbox_inches = 'tight')
        # plt.close()
    