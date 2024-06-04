# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:09:55 2024

@author: Linne
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
import scipy.io
import numpy.ma as ma
import pickle 
import pandas as pd

depth_list = [400, 500, 800]
nlevs = 20

datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
shape = data.temperature.squeeze().shape
depths = data.depth.to_numpy()

depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]

#creating mask for surface (land vs ocean)
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.salinity.squeeze().to_numpy()
volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)
volmask = np.nan_to_num(volume_valid-1, nan = 1)

   
depth_fracs = find_depthfracs(dz, shape, max_depth)

#calculating area represented by each gridpoint
A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)

#creating 3d array of vertical distance represented by each grid point
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]
V_ijk = A_ij3 * dz3
    
depth_i = np.zeros(len(depth_list), dtype = np.int8)
a = 0
for i in depth_list:
    depth_diff = np.abs(depths - i)
    depth_i[a] = np.where(depth_diff == np.min(depth_diff))[0]
    a+=1

extent = [data.lon[0], data.lon[-1], data.lat[0], data.lat[-1]]

# for year in range(startyear, endyear+1):
for year in [2019]:
    # for month in range(1, 13):
    for month in [1]:
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = f'APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(datapath + file)/V_ijk
        
     #plotting multiple onto same figure
     # fig, axs = plt.subplots(len(depth_list), figsize = (12, 30))
     # plt.suptitle(f'WOCE {method} Mean: Single Depth log APE Densities')
 
        for i in range(len(depth_list)):
            fig, ax = plt.subplots(figsize = (16, 12))
            APE_depth = APE_all[depth_i[i], :, :]
            log_APE = ma.masked_array(np.log10(APE_depth), mask = volmask[depth_i[i], :, :]).filled(np.nan)
            #filling with 0 where value <0 (residual from negative APE values)
            negative = np.where(log_APE<0)
            log_APE[negative] = 0
       
            # ax = axs[i]
            ax.set_facecolor('darkgrey')
            plot = ax.imshow(np.flip(log_APE, axis = 0), cmap = 'coolwarm', 
                          extent = extent)
            ax.contour(data.lon, data.lat, log_APE, nlevs, colors = 'black')
       
            plt.colorbar(plot, label = '$log_{10}$ APE density $(Jm^{-3})$', location = 'bottom')
            ax.set_title(f'EN4 {year}-{month} log10 APE Density, Depth: {depths[depth_i[i]]} m')
            plt.savefig(f'EN4 Plots\log10_depth{depth_list[i]}.png', bbox_inches = 'tight')
        # plt.close()
        
        