# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:13:29 2024

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

datadir = datapath + 'WOA-2009'
data = xr.open_dataset(f'{datadir}/salinity_annual_1deg.nc', decode_times = False)
shape = data.s_an.squeeze().shape

#creating 3d bool array for ocean (1) and no ocean (nan)
volume_s = data.s_an.squeeze().to_numpy()
vol_nans = volume_s/volume_s

depths = data.depth.to_numpy()

file = f'{datadir}/WOA_APE.npy'
APE_all = np.load(file)
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
ax.set_title(f'WOA 2009 Zonal Mean')
ax.set_ylabel('Depth, m')
ax.set_xlabel('Latitude, $^\circ$')
ax.invert_yaxis()
plt.savefig(f'WOA Plots\WOA_zonalmean_log10APE.png', bbox_inches = 'tight')
# plt.close()
