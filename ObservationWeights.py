# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:51:31 2024

@author: Linne
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from FuncsAPE import *
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point



datadir = datapath + 'Data' 
#data file has all monthly files inside (no subfolders)
#nothing else in the data file

filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
start_years = [1995, 2005]
end_years = [1999, 2009]

extent = [data.lon[0], data.lon[-1], data.lat[0], data.lat[-1]]
# shape = data.salinity.squeeze().shape
# #converting lat, z into same shape as data
# lat1D = data.lat.to_numpy()
# lat = np.zeros(shape)
# for i in range(len(lat1D)):
#     lat[:, i, :] = lat1D[i]

# z1D = data.depth.to_numpy()
# z = np.zeros(shape)
# for i in range(len(z1D)):
#     z[i, :, :] = z1D[i]

for depth_i in [20, 24]:
# depth_i = 20 #24:800m, 20: 373m
# for depth_i in [24]:
    fig, axs = plt.subplots(2, 2, figsize = (6.3, 3),
                            subplot_kw={'projection':ccrs.PlateCarree()})
    for timei in range(2):
        start_year = start_years[timei]
        end_year = end_years[timei]
        
            
        nmonths = (end_year+1-start_year)*12
        OW_allmonthsT = np.zeros((nmonths, len(data.lat), len(data.lon)))*np.nan
        OW_allmonthsS = np.zeros((nmonths, len(data.lat), len(data.lon)))*np.nan
        i = 0
        for year in range(start_year, end_year+1):
            print(year)
            start =time.time()
            for month in range(1, 13):
                if len(str(month)) == 1:
                    #eg '1' becomes '01' (as in the filenames)
                    month = '0'+str(month)
                filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
                data = xr.open_dataset(f'{datadir}/{filename}')
                OW_allmonthsT[i, :, :] = data.temperature_observation_weights.to_numpy().squeeze()[depth_i, :, :]
                OW_allmonthsS[i, :, :] = data.salinity_observation_weights.to_numpy().squeeze()[depth_i, :, :]
        
                i+=1
        OW_allmonthsT, lon_T = add_cyclic_point(OW_allmonthsT, coord=data.lon, axis=-1)
        OW_allmonthsS, lon_S = add_cyclic_point(OW_allmonthsS, coord=data.lon, axis=-1)
        
        extent = [lon_S[0], lon_S[-1], data.lat[0], data.lat[-1]]

        plot = axs[timei, 0].imshow(np.flip(np.mean(OW_allmonthsT, axis = 0), axis = 0), extent = extent, vmax = 1, vmin = 0)
        axs[timei, 1].imshow(np.flip(np.mean(OW_allmonthsS, axis = 0), axis = 0), extent = extent, vmax = 1, vmin = 0)
        axs[timei, 0].coastlines()
        axs[timei, 1].coastlines()
        
        # axs[timei, 0].contourf(lon_T, data.lat, np.mean(OW_allmonthsT, axis = 0))
        # axs[timei, 1].contourf(lon_S, data.lat, np.mean(OW_allmonthsS, axis = 0))
    
    labels = ['(a)', '(b)', '(c)', '(d)']
    for i in range(4):
        ax = axs.flatten()[i]
        ax.set_title(f'${labels[i]}$', y = - 0.30, fontsize = 9)
    ax = axs[:]
    fig.colorbar(plot, location = 'right', ax = ax, label = 'Observation Weight')
    plt.savefig(f'OW_{depth_i}.pdf', bbox_inches = 'tight')

#%%
