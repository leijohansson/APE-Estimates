# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:00:51 2024

@author: Linne

'Time series' (Jan-Dec) of WOCE APE climatologies
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
#grid spacing 0.25 deg
import scipy.io
import numpy.ma as ma
import pickle 

max_depth = 700

datadir = 'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)
shape = data.temperature.squeeze().shape

depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = depths[1]/2
depth_sum = dz[0]
for i in range(1, len(depths)):
   dz_i = (depths[i]- depth_sum)*2
   dz[i] = dz_i
   depth_sum += dz_i
   
depth_fracs = find_depthfracs(dz, shape, max_depth)

plt.figure()
plt.title('APE Climatology')

months = np.zeros(12)
for i in range(12):
    month = i+1
    if len(str(month)) == 1:
        #eg '1' becomes '01' (as in the filenames)
        month = '0'+str(month)
    months[i] = month


for method in ['BAR', 'PYC']:
    APE_700= np.zeros(12)
    timeidx = -1
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        APE_700[timeidx]= np.sum(APE_all*depth_fracs)
    plt.plot(months, APE_700, label = method)

plt.xlabel('Month')
plt.ylabel('Volume Integrated APE, J')
plt.legend()
plt.savefig('WOCE Plots/APE_Climatology_lineplot.png', bbox_inches = 'tight')

#calculating and plotting volume integrated APE for different ocean basins
with open('RegionFilters/ocean_filters-WOCE.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)

dictkey = list(ocean_filters.keys())[0]
ocean_filters['World'] = np.ones(ocean_filters[dictkey].shape)

n_months = 12

TS_oceans = {}
for OB in ocean_filters.keys():
    TS_oceans[OB] = np.zeros((n_months))

fig, axs = plt.subplots(4, 2, figsize=(12, 15), sharex = True)

for method in ['BAR', 'PYC']:
    time_id = -1
    TS_oceans = {}
    for OB in ocean_filters.keys():
        TS_oceans[OB] = np.zeros((n_months))
        
    for month in range(1, 13):
        time_id += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        APE_700 = np.sum(APE_all*depth_fracs, axis = 0)
        
        for OB in ocean_filters:
            APE_ocean = np.nansum(APE_700 * ocean_filters[OB])
            TS_oceans[OB][time_id] = APE_ocean


    f_i = 0
    for OB in TS_oceans.keys():
        axs[f_i//2, f_i%2].plot(months, TS_oceans[OB], label = method)
        axs[f_i//2, f_i%2].set_title(OB)
        axs[f_i//2, f_i%2].set_ylabel('Volume Integrated APE, $Jm^{-3}$')
    
        f_i += 1
    
axs[3, 0].set_xlabel('Month')
axs[3, 1].set_xlabel('Month')
axs[0, 1].legend(loc = 'upper right', bbox_to_anchor=(1.3, 1), fontsize = 12)
fig.suptitle(f'Volume Integrated APE, depth < {max_depth}m')
fig.savefig('WOCE Plots/Ocean_Climatologies.pdf', bbox_inches = 'tight')
        
        