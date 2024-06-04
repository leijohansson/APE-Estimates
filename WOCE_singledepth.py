# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:47:37 2024

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

depth_list = [400, 500, 800]
nlevs = 20

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
lat = data.latitude
lon = data.longitude

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.salinity.squeeze().to_numpy()
volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)

#finding vertical distance represented by each grid point
depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = depths[1]/2
depth_sum = dz[0]
for i in range(1, len(depths)):
   dz_i = (depths[i]- depth_sum)*2
   dz[i] = dz_i
   depth_sum += dz_i

depth_i = np.zeros(len(depth_list), dtype = np.int8)
a = 0
for i in depth_list:
    depth_diff = np.abs(depths - i)
    depth_i[a] = np.where(depth_diff == np.min(depth_diff))[0]
    a+=1

#calculating area represented by each gridpoint
A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)

#creating 3d array of vertical distance represented by each grid point
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]
    
for method in ['BAR', 'PYC']:
    #loading mean APE file
    filename = f'WOCE_Data/APEarrays/WAGHC_APE_{method}-mean.npy'
    
    #reading APE density
    mean_APE = np.load(datapath + filename)

    #plotting multiple onto same figure
    # fig, axs = plt.subplots(len(depth_list), figsize = (12, 30))
    # plt.suptitle(f'WOCE {method} Mean: Single Depth log APE Densities')
    
    extent = [data.longitude[0], data.longitude[-1], data.latitude[0], data.latitude[-1]]
    for i in range(len(depth_list)):
        fig, ax = plt.subplots(figsize = (16, 12))
        APE_depth = mean_APE[depth_i[i], :, :]
        log_APE = ma.masked_array(np.log10(APE_depth), mask = mask).filled(np.nan)
        #filling with 0 where value <0 (residual from negative APE values)
        negative = np.where(log_APE<0)
        log_APE[negative] = 0

        # ax = axs[i]
        ax.set_facecolor('darkgrey')
        plot = ax.imshow(np.flip(log_APE, axis = 0), cmap = 'coolwarm', vmin = 4.5, 
                      extent = extent)
        ax.contour(lon, lat, log_APE, nlevs, colors = 'black')

        plt.colorbar(plot, label = '$log_{10}$ APE density $(Jm^{-3})$', location = 'bottom')
        ax.set_title(f'WOCE {method} log10 APE Density, Depth: {depths[depth_i[i]]} m')
        # plt.savefig(f'WOCE Plots\{method}\log10_depth{depth_list[i]}.png', bbox_inches = 'tight')
    # plt.close()
    
    