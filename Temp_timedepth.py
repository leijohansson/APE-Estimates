# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:23:52 2024

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

asfrac = False
SO_cutoff_lat = -45

with open(f'RegionFilters/ocean_filters-EN4_{SO_cutoff_lat}.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
ocean_filters['World'] = np.ones(ocean_filters['Indian Ocean'].shape)

with open(f'RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    IO45 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
    IO30 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
IO_band = IO45-IO30
ocean_filters['IO Band'] = IO_band

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
for OB in ['Arctic Ocean']:
    OF = ocean_filters[OB]
    valid_points = data.salinity.to_numpy().squeeze()/data.salinity.to_numpy().squeeze()*OF
    temps = np.zeros((len(data.depth), nmonths))*np.nan
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
            temp = data.temperature.to_numpy().squeeze()
            #calculating APE up to depth 700
            area_temp = A_3d * temp * OF #temp takes care of valid area
            
            onemonth = np.nansum(area_temp, axis = (1, 2))/depth_area
            temps[:, timeidx] = onemonth
    
    mean_temps= np.nanmean(temps, axis = 1)
    mean_temps_2d = np.zeros(temps.shape)
    for i in range(len(mean_temps)):
        mean_temps_2d[i, :] = mean_temps[i]
        
    temps_diff = (temps - mean_temps_2d)
    
    maxval = max(np.abs(np.nanmin(temps_diff)), np.nanmax(temps_diff))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
    
    fig, ax = plt.subplots(figsize = (3.5, 3), layout = 'constrained')
    # fig.suptitle(OB)
    # ax.set_title('Variation in Area averaged salinity with depth and time')
    plot = ax.contourf(x_time, data.depth, temps_diff, cmap = 'seismic', norm = norm)
    ax.set_yscale('log')
    plt.colorbar(plot, label = r"$\theta'$", location = 'bottom')
    ax.set_ylabel('Depth')
    ax.set_xlabel('Time')
    ax.set_ylim(ax.get_ylim()[::-1])
    
    if OB == 'Arctic Ocean':
        # ax.vlines(pd.date_range(f'1990-01-01', periods=1, freq='m'), 0, 4000, color = 'black', linestyle = ':', alpha = 0.7)
        ax.axvline(pd.date_range(f'2016-09-01', periods=1, freq='m'), color = 'black', linestyle = (0, (1, 10)), alpha = 0.7, label ='2016-09')
        ax.axvline(pd.date_range(f'2014-01-01', periods=1, freq='m'), color = 'black', linestyle = (0, (5, 10)), alpha = 0.7, label ='2014-01')
        ax.axvline(pd.date_range(f'1990-10-01', periods=1, freq='m'), color = 'black', linestyle = (0, (3, 10, 1, 10, 1, 10)), alpha = 0.7, label ='1990-10')
        ax.legend(loc = 'lower left')
    if OB == 'Southern Ocean':
        ax.axvline(pd.date_range(f'2006-06-01', periods=1, freq='m'), color = 'black', linestyle = (0, (5, 10)), alpha = 0.7, label ='2006-06')
        ax.legend(loc = 'lower left')

        
    fig.savefig(f'EN4 Plots/TimeVsDepth/Temp/{OB}_TimeVsDepth_{SO_cutoff_lat}.pdf',
                bbox_inches = 'tight')
    
    plt.close()