# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:33:44 2024

@author: Linne
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from FuncsAPE import datapath, calc_Aij
from matplotlib.colors import TwoSlopeNorm
import matplotlib.dates as mdates

asfrac = False
SO_cutoff_lat = -30

with open(f'RegionFilters/ocean_filters-EN4_{SO_cutoff_lat}.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
ocean_filters['World'] = np.ones(ocean_filters['Indian Ocean'].shape)


datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
shape = data.salinity.squeeze().shape

A_ij = calc_Aij(data)
A_3d = np.broadcast_to(A_ij, data.salinity.to_numpy().squeeze().shape)

startyear = 1960
endyear = 2023
nmonths = (endyear + 1 - startyear)*12
x_time =  pd.date_range(f'{startyear}-01-01', periods=nmonths, freq='m')

    
# for OB in ocean_filters.keys():
for OB in ['World']:
    OF = ocean_filters[OB]
    valid_points = data.salinity.to_numpy().squeeze()/data.salinity.to_numpy().squeeze()*OF
    temps = np.zeros((len(data.depth), nmonths))*np.nan
    sals = temps.copy()
    area_valid = A_3d * OF
    depth_area = np.nansum(area_valid, axis = (1, 2))
    timeidx = -1
    for year in range(startyear, endyear+1):
        for month in range(1, 13):
            timeidx += 1
            if len(str(month)) == 1:
                    #eg '1' becomes '01' (as in the filenames)
                month = '0'+str(month)
        
            filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
            data = xr.open_dataset(f'{datadir}/{filename}')
            temp_OW = data.temperature_observation_weights.to_numpy().squeeze()
            sal_OW = data.salinity_observation_weights.to_numpy().squeeze()

            #calculating APE up to depth 700
            # area_temp = A_3d * temp_OW * OF #temp takes care of valid area
            
            onemonth = np.nanmean(temp_OW, axis = (1, 2))#/depth_area
            temps[:, timeidx] = onemonth
            
            onemonthS = np.nanmean(sal_OW, axis = (1, 2))#/depth_area
            sals[:, timeidx] = onemonthS
    #%%
    # norm = TwoSlopeNorm(vcenter = 0.5, vmin = 0, vmax = 0.90)
    fig, axs = plt.subplots(1, 2, figsize = (6.3, 2), sharey = True)
    # fig.suptitle(OB)
    # axs[0].set_title('mean area weighted temperature observation weight with depth and time')
    plot = axs[0].contourf(x_time, data.depth, temps, levels=10, vmin=0.0, vmax=1.0)#, norm = norm)
    
    # ax.set_yscale('log')
    axs[0].set_ylabel('Depth')
    axs[0].set_xlabel('Time')
    axs[0].set_ylim(axs[0].get_ylim()[::-1])
    
    axs[0].set_title('$(a)$', y = -0.45)
    
    axs[0].xaxis.set_minor_locator(mdates.YearLocator(10))
    axs[0].xaxis.set_major_locator(mdates.YearLocator(20))
    



    plot = axs[1].contourf(x_time, data.depth, sals, levels=10, vmin=0.0, vmax=1.0)#, norm = norm)
    # ax.set_yscale('log')
    # plt.colorbar(plot, label = r"Salinity Observation Weight", location = 'bottom')
    axs[1].set_xlabel('Time')
    axs[1].set_title('$(b)$', y = -0.45, fontsize = 10)
    axs[1].xaxis.set_minor_locator(mdates.YearLocator(10))
    axs[1].xaxis.set_major_locator(mdates.YearLocator(20))

    plt.subplots_adjust(wspace=0.1)

    plt.colorbar(plot, location = 'right', ax=axs, label = 'Observation Weight')
    
    fig.savefig('EN4 Plots/TimeVsDepth/OW_TimeVsDepth.pdf',
                bbox_inches = 'tight')

    
    # plt.close()