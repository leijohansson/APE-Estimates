# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:30:23 2024

@author: Linne
"""

import xarray as xr
import numpy as np
from FuncsAPE import *
import scipy.io
import numpy.ma as ma
import pickle 
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def WOCE_singledepth_time(depth, method, density = True):
    '''
    Function to create an 3d array of (time, lat, lon) of APE density (either
    in J/m3 or J/kg) at the gridpoint closest to the selected depth for EN4 
    data

    Parameters
    ----------
    depth : float
        depth for which APE is to be taken at (closest gridpoint).
    method : str
        PYC or BAR
        end year.
    density : bool, optional
        If density is True, take APE density in J/kg. The default is True.

    Returns
    -------
    TS_array : 3D array
        Array of APE density (12, lat, lon). Axis 0 is time, months 
    lon : 1d array
        Array of corresponding longitudes.
    lat : 1d array
        Array of corresponding latitudes.
    depth_true : float
        Depth of the grid point used  (depth of grid point closest to the 
        inputted depth).

    '''
    #reading in sample data to get dataset characteristics
    datadir = datapath + 'WOCE_Data/Data/'
    filename = f'WAGHC_BAR_01_UHAM-ICDC_v1_0_1.nc'
    data = xr.open_dataset(f'{datadir}/{filename}')
    shape = data.temperature.squeeze().shape
    depths = data.depth.to_numpy()
    
    lon = data.longitude.to_numpy()
    lat = data.latitude.to_numpy()
    
    #getting index of depth closest to selected depth
    depth_diff = np.abs(depths - depth)
    depth_i = np.where(depth_diff == np.min(depth_diff))[0]
    if len(depth_i)>0:
        depth_i = depth_i[0]
    depth_true = depths [depth_i]
    
    #creating volume array for calculating density at selected depth
    # A_ij = calc_Aij(data)
    # depth_bnds = data.depth_bnds.to_numpy()
    # dz = depth_bnds[:, 1] - depth_bnds[:, 0]
    # V = A_ij * dz[depth_i]
    
    #creating 3d bool array for ocean (1) and no ocean (0)
    volume_s = data.salinity.squeeze().to_numpy()
    volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)
    mask = np.nan_to_num(volume_valid-1, nan = 1)[depth_i, :, :]

    
    #creating array for data
    TS_array = np.zeros((12, shape[1], shape[2]))
    time_i = 0
    #loop throughtime
    for month in range(1, 13):
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
        
        #reading in APE at depth and converting to density
        if density:
            file = f'{datapath}/WOCE_Data/APEarrays/WAGHC_APE_density_{method}-{month}.npy'
            APE_density = np.load(file)[depth_i, :, :] #APE in J/kg
        else:
            file = f'{datapath}/WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
            APE_density = np.load(file)[depth_i, :, :]/V #APE in Jm^-3
            
        TS_array[time_i, :, :] = ma.masked_array(APE_density,
                                                 mask = mask).filled(np.nan)
            
        time_i += 1
    return TS_array, lon,  lat, depth_true


#%%
def plot_months(depth, method):
    TS_array, lon, lat, depth_true = WOCE_singledepth_time(depth, method)
    TS_array[np.where(TS_array <= 0)] = 10**-10
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    log10 = np.log10(TS_array)
    fig, axs = plt.subplots(4, 3, figsize = (14, 10),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    i = 0
    for ax in axs.flatten():
        # ax.contourf(lon, lat, log10[i, :, :])
        plot = ax.imshow(np.flip(log10[i, :, :], axis = 0), 
                         extent = extent, vmax = np.nanmax(log10), vmin = -4)
        ax.coastlines()
        month_str = str(i+1)
        if len(month_str) == 1:
            month_str = '0'+month_str
        ax.set_title(month_str)
        i += 1
    fig.colorbar(plot, ax=axs, location = 'bottom', fraction = 0.06, pad = 0.04,
                 label = '$log_{10}$(APE density [J/kg])')
    
    fig.suptitle(f'WOCE {method} Climatology: log10 APE density at depth: {depth}m')

def plot_mean(depth, method, ax = False):
    TS_array, lon, lat, depth_true = WOCE_singledepth_time(depth, method)
    TS_array[np.where(TS_array <= 0)] = 10**-10
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    log10 = np.log10(TS_array)
    if not ax:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    plot = ax.imshow(np.flip(np.nanmean(log10, axis = 0), axis = 0), 
                     extent = extent, vmax = 2, vmin = -4)
    ax.coastlines()
    if not ax:
        fig.colorbar(plot, ax=ax, location = 'bottom', fraction = 0.06, pad = 0.04,
                     label = '$log_{10}$(APE density [J/kg])', extend = 'min')
        
        ax.set_title(f'WOCE {method} Climatology: mean log10 APE density at depth: {depth}m')
    else:
        ax.set_title(f'{depth}m')
    return plot

method = 'PYC'
fig, axs = plt.subplots(4, 2, subplot_kw = {'projection': ccrs.PlateCarree()},  figsize = (11, 19))
i = 0
for depth in [0, 100, 200, 300, 400, 600, 800, 1000]:
    plot = plot_mean(depth, method, ax = axs.flatten()[i])
    i += 1
    # plt.close()
fig.suptitle(f'WOCE {method} Climatology: Annual Mean APE Density at Various Depths', y = 0.90)
fig.colorbar(plot, ax=axs, location = 'bottom', fraction = 0.06, pad = 0.04,
             label = '$log_{10}$(APE density [J/kg])', extend = 'min')
# plt.savefig(f'WOCE Plots/log10_depths_{method}.png', bbox_inches = 'tight')

    
   #%% 
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
def tilted_heatmap_in_3d(arr, z, cmap=plt.cm.RdYlGn, ax=None):
    ##Paul Brodersen on StackOverflow
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    for ii, row in enumerate(arr):
        for jj, value in enumerate(row):
            r = Rectangle((ii-0.5, jj-0.5), 1, 1, color=cmap(value))
            ax.add_patch(r)
            art3d.pathpatch_2d_to_3d(r, z=z, zdir="z")

    ax.set_xlim(-1, ii+1)
    ax.set_ylim(-1, jj+1)
    ax.set_zlim(0, 2*z)
    ax.get_figure().canvas.draw()
    
    
def plot_3d(depths, method):
    TS_array, lon, lat, depth_true = WOCE_singledepth_time(depths[0], method)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(lon, lat)
    log10_depths = np.zeros((len(depths), len(lat), len(lon)))
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    for i in range(len(depths)):
        TS_array, lon, lat, depth_true = WOCE_singledepth_time(depths[i], method)
        TS_array[np.where(TS_array <= 0)] = 10**-10
        log10_depths[i, :, :] = np.log10(np.nanmean(TS_array, axis = 0))
        tilted_heatmap_in_3d(log10_depths[i, :, :], -depths[i], ax = ax)

plot_3d([0, 200, 400], 'BAR')