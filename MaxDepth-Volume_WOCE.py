# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:47:58 2024

@author: Linne

Plot of deepest depth at each lat lon point for the WOCE dataset
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
volume_valid = volume_s/volume_s

A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)


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
    
volumes = dz3*A_ij3*volume_valid
vertically_integrated = np.nansum(volumes, axis = 0)
fig, ax = plt.subplots()
ax.set_facecolor('darkgrey')
extent = [data.longitude[0], data.longitude[-1], data.latitude[0], data.latitude[-1]]
plot = ax.imshow(np.flip(vertically_integrated*surface_valid, axis = 0), extent = extent)
plt.colorbar(plot, ax=ax, label = 'Volume covered by grid box (0.25x0.25), $m^3$', location = 'bottom')
fig.savefig('WOCE Plots/VolumePerGridBox.png', bbox_inches = 'tight')


depths_3d = np.zeros(shape)
for i in range(len(depths)):
    depths_3d[i, :, :] = depths[i]
    
depths_3d *= volume_valid
maxdepth = np.nanmax(depths_3d, axis = 0)
fig, ax = plt.subplots()
ax.set_facecolor('darkgrey')
plot = ax.imshow(np.flip(maxdepth, axis = 0), extent = extent)
plt.colorbar(plot, ax=ax, label = 'Deepest Depth, $m$', location = 'bottom')
fig.savefig('WOCE Plots/MaxDepths.png', bbox_inches = 'tight')
