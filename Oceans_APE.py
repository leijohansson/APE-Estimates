# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:34:18 2024

@author: Linne

Plot time series of volume integrated APE for different ocean basins
using EN4 data.
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import pickle 
import matplotlib.dates as mdates
from FuncsAPE import find_depthfracs, datapath

#set max depth to take
max_depth = np.inf

#loading filters
with open('RegionFilters/ocean_filters-EN4.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)

dictkey = list(ocean_filters.keys())[0]
ocean_filters['World'] = np.ones(ocean_filters[dictkey].shape)

#set time range
startyear = 1960
endyear = 1980

n_months = (endyear + 1 - startyear)*12
x_time =  pd.date_range(f'{startyear}-01-01', periods=n_months, freq='m')

#making array to account for maximum depth
datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]

depth_fracs = find_depthfracs(dz, data.salinity.squeeze().shape, max_depth)


#creating arrays to put time series data into
TS_oceans = {}
for OB in ocean_filters.keys():
    TS_oceans[OB] = np.zeros((n_months))


#Calculating volume integrated APE for each ocean basin at each time step
time_id = -1
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
        
        for OB in ocean_filters:
            APE_ocean = np.nansum(APE_700 * ocean_filters[OB])
            TS_oceans[OB][time_id] = APE_ocean
        

#Plots
fig, axs = plt.subplots(4, 2, figsize=(12, 15))
f_i = 0
for OB in TS_oceans.keys():
    axs[f_i//2, f_i%2].plot(x_time, TS_oceans[OB])
    axs[f_i//2, f_i%2].set_title(OB)
    axs[f_i//2, f_i%2].set_ylabel('Volume Integrated APE, $Jm^{-3}$')
    f_i += 1
    

axs[3, 0].set_xlabel('Time')
axs[3, 1].set_xlabel('Time')
fig.suptitle(f'Volume Integrated APE, depths < {max_depth}')
fig.savefig('EN4 Plots/Ocean_APE_TS.pdf', bbox_inches = 'tight')
        
