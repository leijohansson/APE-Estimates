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
import cartopy.crs as ccrs
max_depth = np.inf

datadir = datapath + 'WOCE_Data/Data/'
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
vmin = 11.5
vmax = 14
fs = 20
for method in ['BAR', 'PYC']:
    timespace_arr = np.zeros((12, len(data.latitude), len(data.longitude)))
    timeidx = -1
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = datapath + f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
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
    fig.savefig(f'WOCE Plots/{method}/{method}-meanAPE-{max_depth}m.png', 
                bbox_inches = 'tight')
    plt.close()
    
    #finding anomalies from mean
    anomalies = timespace_arr - mean
    
    pos_anom = np.ones(anomalies.shape)*np.nan
    neg_anom = pos_anom.copy()

    pos_anom[np.where(anomalies > 0)] = anomalies[np.where(anomalies > 0)]
    pos_anom[np.where(anomalies < 0)] = np.nan

    neg_anom[np.where(anomalies < 0)] = anomalies[np.where(anomalies < 0)]
    neg_anom[np.where(anomalies > 0)] = np.nan
        
    #%%
    fig, axs = plt.subplots(3, 4, sharex = True, sharey = True, 
                            figsize = (26, 14), 
                            subplot_kw = {'projection':ccrs.PlateCarree()})
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 
              'Nov', 'Dec']
    fig.tight_layout()
    for month_i in range(anomalies.shape[0]):
        posAPE = ma.masked_array(pos_anom[month_i, :, :], mask = mask)
        logpos = np.log10(posAPE)
        
        
        negAPE = ma.masked_array(neg_anom[month_i, :, :], mask = mask)
        logneg = np.log10(-negAPE)
        
        mAPE = ma.masked_array(anomalies[month_i, :, :], mask = mask)


        if len(str(month_i+1)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month_i+1)
        else:
            month = str(month_i+1)
        #plot and save anomalies from mean
        ax = axs[month_i//4, month_i%4]
        mask_nan = mask.copy()
        mask_nan[np.where(mask_nan) == 0] = np.nan
        ax.imshow(np.flip(mask_nan+1, axis = 0), extent = extent, cmap = 'Greys', alpha = 0.4)
        ax.coastlines()
        ax.set_facecolor('white')
        extent = [data.longitude[0], data.longitude[-1], data.latitude[0], data.latitude[-1]]
        posplot = ax.imshow(np.flip(logpos, axis = 0), extent = extent
                         , cmap = 'Blues', vmin = vmin, vmax = vmax)
        #, norm=colors.CenteredNorm())#, norm=colors.SymLogNorm(linthresh = 1, vmin=-4e15, vmax=4e15, base=1e14))
        negplot = ax.imshow(np.flip(logneg, axis = 0), 
                         extent = extent, cmap = 'Reds', vmin = vmin, vmax = vmax)#, norm=colors.CenteredNorm())#, norm=colors.SymLogNorm(linthresh = 1, vmin=-4e15, vmax=4e15, base=1e14))
        # ax.set_title(f'WOCE APE anomaly 700m {method}-{month}: {np.round(np.nansum(mAPE), -15)}J')
        ax.set_title(f"{months[month_i]}", fontsize = fs)#, $\sum APE' = {np.round(np.nansum(mAPE), -16)}$J")

    ax0 = axs[:, :2]
    ax1 = axs[:, 2:]

    cb1 = fig.colorbar(posplot, location = 'bottom', ax = ax1, extend = 'min',
                 pad = 0.02, aspect = 30)
    cb1.set_label(label = "$log_{10}$ (positive APE')", fontsize = fs)
    cb1.ax.tick_params(labelsize=18)    

    cb2 = fig.colorbar(negplot, location = 'bottom', ax = ax0, extend = 'min',
                 pad = 0.02, aspect = 30)
    cb2.set_label(label = "$log_{10}$ (negative APE')", fontsize = fs)
    cb2.ax.tick_params(labelsize=18)    

    fig.suptitle(f'Vertically Integrated APE Anomalies: WOCE {method}', y = 0.95, fontsize = fs)
    fig.savefig(f'WOCE Plots/{method}/Anomalies/{method}-{max_depth}m_anomalies.png', 
                bbox_inches = 'tight')
    # plt.close()

