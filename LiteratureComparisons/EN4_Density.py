# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:00:24 2024

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

max_depth = np.inf

datadir = 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
shape = data.temperature.squeeze().shape

depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]



#creating mask for surface (land vs ocean)
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.salinity.squeeze().to_numpy()
volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)
   
depth_fracs = find_depthfracs(dz, shape, max_depth)

#calculating area represented by each gridpoint
A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)

#creating 3d array of vertical distance represented by each grid point
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]
    
# for year in range(startyear, endyear+1):
for year in [2019]:
    # for month in range(1, 13):
    for month in [1]:
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = f'APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(file)
        
        #calculating vertical integral of mean APE (all depths)
        vert_int = np.sum(APE_all, axis = 0)
        
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
        fig, axs = plt.subplots(2, figsize = (12, 12))
        fig.suptitle(f'EN4 {year}-{month}')
        axs[0].set_facecolor('darkgrey')
        extent = [data.lon[0], data.lon[-1], data.lat[0], data.lat[-1]]
        plot = axs[0].imshow(np.flip(log_APEm2, axis = 0), cmap = 'coolwarm', vmin = 4.5, 
                          extent = extent)
        plt.colorbar(plot, ax = axs[0], shrink = 0.7,
                     label = '$log_{10}$ of top to bottom APE $(Jm^{-2})$', location = 'bottom')
        axs[0].set_title('Vertically Integrated')
        # plt.savefig(f'WOCE Plots\{method}\Log10_btt_APE_{method}.png', bbox_inches = 'tight')
        # plt.close()
        
        
        #vertically averaged
        zsum = np.sum(dz3*volume_valid, axis = 0)
        VA_per_area = VI_per_area / zsum
        
        #finding log
        log_APEm2 = np.log10(VA_per_area).filled(np.nan)
        #getting rid of negative values
        negative_m2 = np.where(log_APEm2<0)
        log_APEm2[negative] = 0
        
        #plotting vertically averaged APE density
        axs[1].set_facecolor('darkgrey')
        plot = axs[1].imshow(np.flip(log_APEm2, axis = 0), cmap = 'coolwarm', 
                          extent = extent, vmin = 1)
        plt.colorbar(plot, ax=axs[1], shrink = 0.7,
                     label = '$log_{10}$ of vertically averaged top to bottom APE $(Jm^{-3})$',
                     location = 'bottom')
        # test = axs[1].contour(data.lon, data.lat, log_APEm2, 20, colors = 'black') 
        axs[1].set_title('Vertically Averaged (APE Density)')
        plt.savefig(f'EN4 Plots\VI-VA-comparison.png', bbox_inches = 'tight')
        # plt.close()
        #%%
import scipy.ndimage as ndimage
nlevs = 20
sigma = 0.7
lon = data.lon
lat = data.lat


smoothed = ndimage.gaussian_filter(log_APEm2, sigma=sigma, order=0)
fig, axs = plt.subplots(2, figsize = (12, 12))
fig.tight_layout()
axs[0].set_facecolor('darkgrey')
axs[0].contour(lon, lat, log_APEm2, nlevs, colors = 'black')
axs[0].set_title('Original Data')
axs[0].imshow(np.flip(log_APEm2, axis = 0), cmap = 'coolwarm', 
                  extent = extent, vmin = 1)

axs[1].set_facecolor('darkgrey')
axs[1].set_title(f'Gaussian Filtered Contour, sigma = {sigma}')
axs[1].contour(lon, lat, smoothed, nlevs, colors = 'black')
plot = axs[1].imshow(np.flip(log_APEm2, axis = 0), cmap = 'coolwarm', 
                  extent = extent, vmin = 1)

plt.savefig(f'EN4 Plots/log10density_with_contour_sigma{sigma}.pdf')

# plt.colorbar(plot, ax = axs[2], label = '$log_{10}$ of vertically averaged top to bottom APE $(Jm^{-3})$', location = 'bottom')


