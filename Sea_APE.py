# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:18:31 2024

@author: Linne
Plotting time series of volume integrated APE for depths up to 700m for 
mediterannean sea and the red sea
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import pickle 
import matplotlib.dates as mdates
from FuncsAPE import find_depthfracs

max_depth = 700
startyear = 1960
endyear = 2022


with open('RegionFilters/sea_filters-EN4.pkl', 'rb') as f:
    sea_filters = pickle.load(f)

dictkey = list(sea_filters.keys())[0]
sea_filters['World'] = np.ones(sea_filters[dictkey].shape)

n_months = (endyear + 1 - startyear)*12
x_time =  pd.date_range(f'{startyear}-01-01', periods=n_months, freq='m')


#making array to account for maximum depth
datadir = 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]

depth_fracs = find_depthfracs(dz, data.salinity.squeeze().shape, max_depth)

TS_sea = {}
for sea in sea_filters.keys():
    TS_sea[sea] = np.zeros((n_months))


time_id = -1
#creating time series for different seas
for year in range(startyear, endyear+1):
    for month in range(1, 13):
        time_id += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = f'APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        APE_700 = np.sum(APE_all*depth_fracs, axis = 0)
        
        for sea in sea_filters:
            APE_sea = np.nansum(APE_700 * sea_filters[sea])
            TS_sea[sea][time_id] = APE_sea
        
#plotting time series 
fig, axs = plt.subplots(2, 2, figsize=(12, 15))
f_i = 0
for sea in TS_sea.keys():
    axs[f_i//2, f_i%2].plot(x_time, TS_sea[sea])
    axs[f_i//2, f_i%2].set_title(sea)
    axs[f_i//2, f_i%2].set_ylabel('Volume Integrated APE, $Jm^{-3}$')
    f_i += 1

axs[1, 0].set_xlabel('Time')
axs[1, 1].set_xlabel('Time')
fig.suptitle(f'Volume Integrated APE, depths < {max_depth}m')
fig.savefig('EN4 Plots/Sea_APE-700m_TS.pdf', bbox_inches = 'tight')
        