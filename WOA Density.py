# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:38:04 2024

@author: Linne
"""

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

#maximum depth
depth_co = np.inf

#read data
datadir = datapath + 'WOA-2009'
data = xr.open_dataset(f'{datadir}/salinity_annual_1deg.nc', decode_times = False)
shape = data.s_an.squeeze().shape

#calculate vertical distance between grid points
depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = (depths[1]-depths[0])/2
for i in range(1, len(depths)-1):
    dz[i] = 0.5*(depths[i+1] - depths[i-1])
dz[-1] = 0.5*(depths[-1] - depths[-2])


#creating mask for surface (land vs ocean)
surface_s = data.s_an.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.s_an.squeeze().to_numpy()
volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)
   
depth_fracs = find_depthfracs(dz, shape, max_depth)

#calculating area represented by each gridpoint
A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)

#creating 3d array of vertical distance represented by each grid point
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]
    
            
file = f'{datadir}/WOA_APE.npy'
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
fig, axs = plt.subplots(2, figsize = (12, 18))
fig.suptitle(f'WOA 2009')
axs[0].set_facecolor('darkgrey')
extent = [data.lon[0], data.lon[-1], data.lat[0], data.lat[-1]]
plot = axs[0].imshow(np.flip(log_APEm2, axis = 0), cmap = 'coolwarm', vmin = 4.5, 
                  extent = extent)
test = axs[0].contour(data.lon, data.lat, log_APEm2, 20, colors = 'black') 
plt.colorbar(plot, ax = axs[0], shrink = 0.7,
             label = '$log_{10}$ of top to bottom APE $(Jm^{-2})$', location = 'bottom')
axs[0].set_title('Vertically Integrated')
# plt.savefig(f'WOCE Plots\{method}\Log10_btt_APE_{method}.png', bbox_inches = 'tight')
# plt.close()


#vertically averaged
zsum = np.sum(dz3*volume_valid, axis = 0)
VA_per_area = VI_per_area / zsum

#finding log
log_APEm3 = np.log10(VA_per_area).filled(np.nan)
#getting rid of negative values
negative_m3 = np.where(log_APEm3<0)
log_APEm3[negative] = 0

#plotting vertically averaged APE density
axs[1].set_facecolor('darkgrey')
plot = axs[1].imshow(np.flip(log_APEm3, axis = 0), cmap = 'coolwarm', 
                  extent = extent, vmin = 1)
plt.colorbar(plot, ax=axs[1], shrink = 0.7,
             label = '$log_{10}$ of vertically averaged top to bottom APE $(Jm^{-3})$',
             location = 'bottom')
test = axs[1].contour(data.lon, data.lat, log_APEm3, 20, colors = 'black') 
axs[1].set_title('Vertically Averaged (APE Density)')
plt.savefig(f'WOA Plots\VI-VA-comparison.png', bbox_inches = 'tight')
        # plt.close()
        #%%
import scipy.ndimage as ndimage
nlevs = 30
sigma = 1
lon = data.lon
lat = data.lat


def filter_nan_gaussian_conserving2(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr / (1-loss)
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    return gauss

# smoothed = ndimage.gaussian_filter(log_APEm2, sigma=sigma, order=0)
smoothed = filter_nan_gaussian_conserving2(log_APEm3, sigma)
fig, axs = plt.subplots(2, figsize = (12, 12))
fig.tight_layout()
axs[0].set_facecolor('darkgrey')
axs[0].contour(lon, lat, log_APEm3, nlevs, colors = 'black')
axs[0].set_title('Original Data')
axs[0].imshow(np.flip(log_APEm3, axis = 0), cmap = 'coolwarm', 
                  extent = extent, vmin = 1)

axs[1].set_facecolor('darkgrey')
axs[1].set_title(f'Gaussian Filtered Contour, sigma = {sigma}')
axs[1].contour(lon, lat, smoothed, nlevs, colors = 'black')
plot = axs[1].imshow(np.flip(log_APEm3, axis = 0), cmap = 'coolwarm', 
                  extent = extent, vmin = 1)

# plt.savefig(f'EN4 Plots/log10density_with_contour_sigma{sigma}.pdf')

# plt.colorbar(plot, ax = axs[2], label = '$log_{10}$ of vertically averaged top to bottom APE $(Jm^{-3})$', location = 'bottom')


