# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:40:51 2024

@author: Linne
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from cartopy.util import add_cyclic_point
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

depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]
dz3d = np.zeros(data.salinity.to_numpy().squeeze().shape)
for i in range(len(dz)):
    dz3d[i, :, :] = dz[i]
#%%
# for OB in ocean_filters.keys():
for OB in ['Southern Ocean', 'Arctic Ocean']:
    OF = ocean_filters[OB]
    APE = np.zeros((len(data.depth), nmonths))*np.nan
    
    timeidx = -1
    for year in range(startyear, endyear+1):
        for month in range(1, 13):
            timeidx += 1
            if len(str(month)) == 1:
                #eg '1' becomes '01' (as in the filenames)
                month = '0'+str(month)
    
            file = f'{datapath}APEarrays\APE_{year}-{month}.npy'
            APE_all = np.load(file)
            #calculating APE up to depth 700
            onemonth = np.nansum(APE_all/dz3d*OF, axis = (1, 2))
            APE[:, timeidx] = onemonth
    
    mean_APE = np.mean(APE, axis = 1)
    mean_APE_2d = np.zeros(APE.shape)
    for i in range(len(mean_APE)):
        mean_APE_2d[i, :] = mean_APE[i]
        
    if asfrac:
        APE_diff = (APE - mean_APE_2d)/mean_APE_2d
    else:
        APE_diff = (APE - mean_APE_2d)

    maxval = max(np.abs(np.nanmin(APE_diff)), np.nanmax(APE_diff))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
    
    if ('Atlantic'in OB) or ('Pacific' in OB) and ('Equatorial' not in OB):
        fig, ax = plt.subplots(figsize = (3.5, 3), layout = 'constrained')
        location  = 'bottom'
    else:
        fig, ax = plt.subplots(figsize = (5, 3), layout = 'constrained')
        location = 'right'
    print(location)
    # fig.suptitle(OB)
    # ax.set_title('Variation in Area Integrated APE with depth and time')
    plot = ax.contourf(x_time, data.depth, APE_diff, cmap = 'seismic', norm = norm)
    ax.set_yscale('log')
    if asfrac:
        plt.colorbar(plot, label = r"$\frac{APE'}{\overline{APE}}$", location = 'bottom')
    else:
        plt.colorbar(plot, label = r"$APE', Jm^{-1}$", location = location)
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
    if asfrac:
        fig.savefig(f'EN4 Plots/TimeVsDepth/asfrac/{OB}_TimeVsDepth_{SO_cutoff_lat}.pdf', bbox_inches = 'tight')
    else:
        fig.savefig(f'EN4 Plots/TimeVsDepth/{OB}_TimeVsDepth_{SO_cutoff_lat}.pdf', bbox_inches = 'tight')

    plt.close()