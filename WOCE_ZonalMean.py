# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:35:18 2024

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

max_depth = np.inf
vmin = 1 #minimum log10 value

datadir = datapath + 'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)
shape = data.temperature.squeeze().shape

#creating mask for surface (land vs ocean)
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.salinity.squeeze().to_numpy()
volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)
vol_nans = volume_s/volume_s

#finding vertical distance represented by each grid point
depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = depths[1]/2
depth_sum = dz[0]
for i in range(1, len(depths)):
   dz_i = (depths[i]- depth_sum)*2
   dz[i] = dz_i
   depth_sum += dz_i
   
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]
depth_fracs = find_depthfracs(dz, shape, max_depth)
V = calc_Aij(data)*dz3

for method in ['BAR', 'PYC']:
    #loading mean APE file
    filename = f'WOCE_Data/APEarrays/WAGHC_APE_{method}-mean.npy'
    mean_APE = np.load(datapath + filename)
    mean_APE = mean_APE*vol_nans/V
    zonalmean = np.nanmean(mean_APE, axis = 2)
    log10zm = np.log10(zonalmean)
    flipped = np.flip(log10zm, axis =0)
    #plotting vertically integrated APE per unit area
    fig, ax = plt.subplots(figsize = (6.3, 3))
    ax.set_facecolor('darkgrey')
    X, Y = np.meshgrid(data.latitude, data.depth)
    log10zm[np.where(log10zm < vmin)] = vmin
    plot = ax.contourf(X, Y, log10zm, levels = 15, cmap = 'viridis')
    # plt.gca().flip
    plt.colorbar(plot, label = '$log_{10}$ zonal mean APE density $(Jm^{-3})$', location = 'right', extend = 'min')
    # ax.set_title(f'WOCE {method} Zonal Mean')
    ax.set_ylabel('Depth, m')
    ax.set_xlabel('Latitude, $^\circ$')
    ax.invert_yaxis()
    plt.savefig(f'WOCE Plots\{method}\WOCE_{method}_zonalmean_log10APE.pdf', bbox_inches = 'tight')
    # plt.close()
    
    