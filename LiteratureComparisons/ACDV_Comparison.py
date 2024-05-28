# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:00:24 2024

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

datadir = 'WOCE_Data/Data/'
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

#finding vertical distance represented by each grid point
depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = depths[1]/2
depth_sum = dz[0]
for i in range(1, len(depths)):
   dz_i = (depths[i]- depth_sum)*2
   dz[i] = dz_i
   depth_sum += dz_i

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
    mean_APE = np.load(filename)
    
    #calculating vertical integral of mean APE (all depths)
    vert_int = np.sum(mean_APE, axis = 0)
    
    #masking non ocean areas
    VIm = ma.masked_array(vert_int, mask = mask)
    Am = ma.masked_array(A_ij, mask = mask)
    
    #calculating log10 of vertically integrated APE per unit area
    VI_per_area = vert_int/Am
    log_APEm2 = np.log10(VI_per_area).filled(np.nan)
    #filling with 0 where value <0 (residual from negative APE values)
    negative = np.where(log_APEm2<0)
    log_APEm2[negative] = 0
    
    #plotting vertically integrated APE per unit area
    fig, ax = plt.subplots()
    ax.set_facecolor('darkgrey')
    extent = [data.longitude[0], data.longitude[-1], data.latitude[0], data.latitude[-1]]
    plot = ax.imshow(np.flip(log_APEm2, axis = 0), cmap = 'coolwarm', vmin = 4.5, 
                      extent = extent)
    plt.colorbar(plot, label = '$log_{10}$ of top to bottom APE $(Jm^{-2})$', location = 'bottom')
    plt.title(f'WOCE {method} Mean, Vertically Integrated')
    plt.savefig(f'WOCE Plots\{method}\Log10_btt_APE_{method}.png', bbox_inches = 'tight')
    plt.close()
    
    
    #vertically averaged
    zsum = np.sum(dz3*volume_valid, axis = 0)
    VA_per_area = VI_per_area / zsum
    
    #finding log
    log_APEm2 = np.log10(VA_per_area).filled(np.nan)
    #getting rid of negative values
    negative_m2 = np.where(log_APEm2<0)
    log_APEm2[negative] = 0
    
    #plotting vertically averaged APE density
    fig1, ax1 = plt.subplots()
    ax1.set_facecolor('darkgrey')
    plot = ax1.imshow(np.flip(log_APEm2, axis = 0), cmap = 'coolwarm', 
                      extent = extent, vmin = 1)
    plt.colorbar(plot, label = '$log_{10}$ of vertically averaged top to bottom APE $(Jm^{-3})$', location = 'bottom')
    plt.title(f'WOCE {method} Mean, Vertically Averaged')
    plt.savefig(f'WOCE Plots\{method}\Log10_btt_VA_APE_{method}.png', bbox_inches = 'tight')
    plt.close()
