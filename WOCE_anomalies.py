# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:20:56 2024

@author: Linne

Vertically integrated WOCE APE anomalies
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
#grid spacing 0.25 deg
import scipy.io
import numpy.ma as ma
import pickle 
import matplotlib.colors as colors
max_depth = 700

datadir = 'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)
shape = data.temperature.squeeze().shape

#creating mask to account for topography
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

# finding vertical distance between depths/ dz accounted for by each
# grid point
depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = depths[1]/2
depth_sum = dz[0]
for i in range(1, len(depths)):
   dz_i = (depths[i]- depth_sum)*2
   dz[i] = dz_i
   depth_sum += dz_i
   
#finding depth frac array
depth_fracs = find_depthfracs(dz, shape, max_depth)

for method in ['BAR', 'PYC']:
    timespace_arr = np.zeros((12, len(data.latitude), len(data.longitude)))
    timeidx = -1
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        APE_700 = np.sum(APE_all*depth_fracs, axis = 0)
        #creating array with vertically integrated APE for all months
        timespace_arr[timeidx, :, :] = APE_700
        
    #finding time mean (over year) climatology
    mean = np.mean(timespace_arr, axis = 0)
    mMean = ma.masked_array(APE_700, mask = mask)

    #plotting time mean APE
    fig, ax = plt.subplots()
    ax.set_facecolor('lightgrey')
    extent = [data.longitude[0], data.longitude[-1], data.latitude[0], data.latitude[-1]]
    plot = ax.imshow(np.flip(mMean, axis = 0), extent = extent)
    ax.set_title(f'WOCE APE 700m {method}- Year Mean: {np.round(np.sum(mean), -15)}J')
    plt.colorbar(plot, location = 'bottom')
    ax.set_ylabel('Latitude, $^\circ$')
    ax.set_xlabel('Longitude, $^\circ$')
    fig.savefig(f'WOCE Plots/{method}/{method}-meanAPE-700m.png', 
                bbox_inches = 'tight')
    plt.close()
    
    #finding anomalies from mean
    anomalies = timespace_arr - mean
    
    fig, axs = plt.subplots(3, 4, sharex = True, sharey = True, figsize = (26, 12))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 
              'Nov', 'Dec']
    fig.tight_layout()
    for month_i in range(anomalies.shape[0]):
        mAPE = ma.masked_array(anomalies[month_i, :, :], mask = mask)
        if len(str(month_i+1)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month_i+1)
        else:
            month = str(month_i+1)
        #plot and save anomalies from mean
        # fig, ax = plt.subplots()
        ax = axs[month_i//4, month_i%4]
        ax.set_facecolor('black')
        extent = [data.longitude[0], data.longitude[-1], data.latitude[0], data.latitude[-1]]
        plot = ax.imshow(np.flip(mAPE, axis = 0), extent = extent, cmap = 'bwr', norm=colors.CenteredNorm())#, norm=colors.SymLogNorm(linthresh = 1, vmin=-4e15, vmax=4e15, base=1e14))
        # ax.set_title(f'WOCE APE anomaly 700m {method}-{month}: {np.round(np.nansum(mAPE), -15)}J')
        ax.set_title(f'{months[month_i]}: {np.round(np.nansum(mAPE), -15)}J')
        plt.colorbar(plot, location = 'bottom')
        # ax.set_ylabel('Latitude, $^\circ$')
        # ax.set_xlabel('Longitude, $^\circ$')
        


    # fig.subplots_adjust(bottom=0.15, top = 0.97)
    # plt.subplots_adjust(wspace=0.02, hspace=0.005)
    # cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.02])
    # fig.colorbar(plot, cax=cbar_ax, label = 'Vertically Integrated APE anomaly, $Jm^{-2}$', 
    #              location = 'bottom')

    fig.suptitle(f'WOCE APE anomaly 700m: {method}')
    fig.savefig(f'WOCE Plots/{method}/Anomalies/{method}-700m_anomalies.png', 
                bbox_inches = 'tight')
    plt.close()

